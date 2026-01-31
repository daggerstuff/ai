"""
API Integration Layer for Dataset Training Pipeline
Provides REST API endpoints for web frontend and external integrations
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .dataset_taxonomy import DatasetTaxonomy, DatasetMetadata, DatasetCategory
from .training_styles import TrainingStyleManager, TrainingStyle, BaseTrainingConfig
from .training_orchestrator import TrainingPipelineOrchestrator
from .mcp_integration import DatasetTrainingMCPInterface, MCPTask

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Pydantic models for API requests and responses
class DatasetCategorizeRequest(BaseModel):
    """Request model for dataset categorization"""
    dataset_path: str = Field(..., description="Path to the dataset file")
    dataset_name: str = Field(default="unnamed_dataset", description="Name of the dataset")
    dataset_description: str = Field(default="", description="Description of the dataset")
    sample_size: int = Field(default=100, ge=10, le=1000, description="Sample size for analysis")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata")


class DatasetCategorizeResponse(BaseModel):
    """Response model for dataset categorization"""
    category: str = Field(..., description="Recommended category")
    confidence: float = Field(..., description="Categorization confidence score")
    recommended_subcategories: List[str] = Field(..., description="Recommended subcategories")
    metadata: Dict[str, Any] = Field(..., description="Updated dataset metadata")
    analysis_summary: Dict[str, Any] = Field(..., description="Content analysis summary")


class TrainingStyleSelectRequest(BaseModel):
    """Request model for training style selection"""
    dataset_metadata: Dict[str, Any] = Field(..., description="Dataset metadata")
    training_goals: Dict[str, Any] = Field(default={}, description="Training goals and constraints")
    available_compute: str = Field(default="medium", description="Available compute resources")


class TrainingStyleSelectResponse(BaseModel):
    """Response model for training style selection"""
    selected_style: str = Field(..., description="Selected training style")
    style_info: Dict[str, Any] = Field(..., description="Information about the selected style")
    config_template: Dict[str, Any] = Field(..., description="Configuration template")
    confidence: float = Field(..., description="Selection confidence score")


class TrainingConfigRequest(BaseModel):
    """Request model for training configuration"""
    style: str = Field(..., description="Training style")
    config: Dict[str, Any] = Field(default={}, description="Configuration parameters")
    optimize: bool = Field(default=False, description="Whether to optimize configuration")
    optimization_strategy: str = Field(default="balanced", description="Optimization strategy")


class TrainingConfigResponse(BaseModel):
    """Response model for training configuration"""
    config: Dict[str, Any] = Field(..., description="Training configuration")
    valid: bool = Field(..., description="Whether configuration is valid")
    validation_errors: List[str] = Field(..., description="Validation errors if any")
    optimization_applied: bool = Field(..., description="Whether optimization was applied")


class PipelineExecuteRequest(BaseModel):
    """Request model for pipeline execution"""
    dataset_path: str = Field(..., description="Path to dataset file")
    dataset_name: str = Field(..., description="Name of the dataset")
    dataset_description: str = Field(..., description="Description of the dataset")
    training_objective: str = Field(default="therapeutic_accuracy", description="Training objective")
    available_compute: str = Field(default="medium", description="Available compute resources")
    safety_requirements: str = Field(default="moderate", description="Safety requirements")
    use_container: bool = Field(default=False, description="Whether to use containerized training")
    enable_monitoring: bool = Field(default=True, description="Whether to enable monitoring")


class PipelineExecuteResponse(BaseModel):
    """Response model for pipeline execution"""
    execution_id: str = Field(..., description="Pipeline execution ID")
    status: str = Field(..., description="Execution status")
    pipeline_id: str = Field(..., description="Pipeline ID")
    estimated_duration: str = Field(..., description="Estimated execution duration")


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status"""
    execution_id: str = Field(..., description="Execution ID")
    status: str = Field(..., description="Current status")
    progress: float = Field(..., description="Overall progress percentage")
    current_stage: str = Field(..., description="Current pipeline stage")
    stages: List[Dict[str, Any]] = Field(..., description="Detailed stage information")
    start_time: str = Field(..., description="Execution start time")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")


class TrainingResultsResponse(BaseModel):
    """Response model for training results"""
    execution_id: str = Field(..., description="Execution ID")
    status: str = Field(..., description="Training status")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="F1 score")
    empathy_score: float = Field(..., description="Empathy score")
    therapeutic_appropriateness: float = Field(..., description="Therapeutic appropriateness score")
    safety_score: float = Field(..., description="Safety score")
    model_path: str = Field(..., description="Path to trained model")
    checkpoint_path: str = Field(..., description="Path to training checkpoints")


class DatasetTrainingAPI:
    """API layer for dataset training pipeline"""
    
    def __init__(self, orchestrator: TrainingPipelineOrchestrator):
        self.orchestrator = orchestrator
        self.taxonomy = orchestrator.taxonomy
        self.style_manager = orchestrator.style_manager
        self.mcp_interface = DatasetTrainingMCPInterface(orchestrator)
        self.app = FastAPI(
            title="Dataset Training API",
            description="API for dataset categorization and training style management",
            version="1.0.0"
        )
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "dataset_training_api",
                "version": "1.0.0"
            }
        
        @self.app.post("/api/datasets/categorize", response_model=DatasetCategorizeResponse)
        async def categorize_dataset(request: DatasetCategorizeRequest):
            """Categorize a dataset based on content analysis"""
            try:
                # Create MCP task
                task = MCPTask(
                    task_type="dataset_categorize",
                    parameters={
                        "dataset_path": request.dataset_path,
                        "dataset_name": request.dataset_name,
                        "dataset_description": request.dataset_description,
                        "sample_size": request.sample_size,
                        "metadata": request.metadata
                    }
                )
                
                # Process task
                result = await self.mcp_interface.process_task(task)
                
                if result["status"] == "failed":
                    raise HTTPException(status_code=400, detail=result["error"])
                
                return DatasetCategorizeResponse(
                    category=result["results"]["recommended_category"],
                    confidence=result["results"]["confidence"],
                    recommended_subcategories=result["results"]["recommended_subcategories"],
                    metadata=result["results"]["metadata"],
                    analysis_summary=result["results"]["analysis_summary"]
                )
                
            except Exception as e:
                logger.error(f"Dataset categorization failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/datasets/validate")
        async def validate_dataset(metadata: Dict[str, Any]):
            """Validate dataset metadata"""
            try:
                # Create MCP task
                task = MCPTask(
                    task_type="dataset_validate",
                    parameters={"metadata": metadata}
                )
                
                # Process task
                result = await self.mcp_interface.process_task(task)
                
                if result["status"] == "failed":
                    raise HTTPException(status_code=400, detail=result["error"])
                
                return result["results"]
                
            except Exception as e:
                logger.error(f"Dataset validation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/datasets/recommendations")
        async def get_dataset_recommendations(metadata: Dict[str, Any]):
            """Get recommendations for dataset processing"""
            try:
                # Create MCP task
                task = MCPTask(
                    task_type="dataset_recommendations",
                    parameters={"metadata": metadata}
                )
                
                # Process task
                result = await self.mcp_interface.process_task(task)
                
                if result["status"] == "failed":
                    raise HTTPException(status_code=400, detail=result["error"])
                
                return result["results"]
                
            except Exception as e:
                logger.error(f"Dataset recommendations failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/training/styles/select", response_model=TrainingStyleSelectResponse)
        async def select_training_style(request: TrainingStyleSelectRequest):
            """Select optimal training style for dataset"""
            try:
                # Create MCP task
                task = MCPTask(
                    task_type="training_style_select",
                    parameters={
                        "dataset_metadata": request.dataset_metadata,
                        "training_goals": request.training_goals,
                        "available_compute": request.available_compute
                    }
                )
                
                # Process task
                result = await self.mcp_interface.process_task(task)
                
                if result["status"] == "failed":
                    raise HTTPException(status_code=400, detail=result["error"])
                
                return TrainingStyleSelectResponse(
                    selected_style=result["results"]["selected_style"],
                    style_info=result["results"]["style_info"],
                    config_template=result["results"]["config_template"],
                    confidence=result["results"]["selection_confidence"]
                )
                
            except Exception as e:
                logger.error(f"Training style selection failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/training/styles/configure", response_model=TrainingConfigResponse)
        async def configure_training_style(request: TrainingConfigRequest):
            """Configure training style parameters"""
            try:
                # Create MCP task
                task = MCPTask(
                    task_type="training_style_configure",
                    parameters={
                        "style": request.style,
                        "config": request.config,
                        "optimize": request.optimize,
                        "optimization_strategy": request.optimization_strategy
                    }
                )
                
                # Process task
                result = await self.mcp_interface.process_task(task)
                
                if result["status"] == "failed":
                    raise HTTPException(status_code=400, detail=result["error"])
                
                return TrainingConfigResponse(
                    config=result["results"]["config"],
                    valid=result["results"]["valid"],
                    validation_errors=result["results"]["validation_errors"],
                    optimization_applied=result["results"]["optimization_applied"]
                )
                
            except Exception as e:
                logger.error(f"Training configuration failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/training/execute", response_model=PipelineExecuteResponse)
        async def execute_training_pipeline(
            request: PipelineExecuteRequest,
            background_tasks: BackgroundTasks
        ):
            """Execute complete training pipeline"""
            try:
                # Create pipeline configuration
                pipeline_config = {
                    "dataset_path": request.dataset_path,
                    "dataset_name": request.dataset_name,
                    "dataset_description": request.dataset_description,
                    "training_objective": request.training_objective,
                    "available_compute": request.available_compute,
                    "safety_requirements": request.safety_requirements,
                    "use_container": request.use_container,
                    "enable_monitoring": request.enable_monitoring
                }
                
                # Execute pipeline
                execution_id = await self.orchestrator.execute_pipeline(pipeline_config)
                
                # Start background monitoring
                background_tasks.add_task(self._monitor_execution, execution_id)
                
                return PipelineExecuteResponse(
                    execution_id=execution_id,
                    status="started",
                    pipeline_id=execution_id,
                    estimated_duration="2-4 hours"  # This could be calculated
                )
                
            except Exception as e:
                logger.error(f"Pipeline execution failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/training/{execution_id}/status", response_model=PipelineStatusResponse)
        async def get_training_status(execution_id: str):
            """Get training pipeline status"""
            try:
                status = self.orchestrator.get_execution_status(execution_id)
                
                if not status:
                    raise HTTPException(status_code=404, detail="Execution not found")
                
                return PipelineStatusResponse(
                    execution_id=status["execution_id"],
                    status=status["status"],
                    progress=status["progress"],
                    current_stage=status["current_stage"],
                    stages=status["stages"],
                    start_time=status["start_time"],
                    estimated_completion=self._estimate_completion(status)
                )
                
            except Exception as e:
                logger.error(f"Status retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/training/{execution_id}/results", response_model=TrainingResultsResponse)
        async def get_training_results(execution_id: str):
            """Get training results"""
            try:
                # Create MCP task
                task = MCPTask(
                    task_type="training_results",
                    parameters={"execution_id": execution_id}
                )
                
                # Process task
                result = await self.mcp_interface.process_task(task)
                
                if result["status"] == "failed":
                    raise HTTPException(status_code=400, detail=result["error"])
                
                results = result["results"]
                
                return TrainingResultsResponse(
                    execution_id=results["execution_id"],
                    status=results["status"],
                    accuracy=results["accuracy"],
                    precision=results["precision"],
                    recall=results["recall"],
                    f1_score=results["f1_score"],
                    empathy_score=results["empathy_score"],
                    therapeutic_appropriateness=results["therapeutic_appropriateness"],
                    safety_score=results["safety_score"],
                    model_path=results["model_path"],
                    checkpoint_path=results["checkpoint_path"]
                )
                
            except Exception as e:
                logger.error(f"Results retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/api/training/{execution_id}")
        async def cancel_training_pipeline(execution_id: str):
            """Cancel training pipeline execution"""
            try:
                success = await self.orchestrator.cancel_execution(execution_id)
                
                if not success:
                    raise HTTPException(status_code=404, detail="Execution not found or cannot be cancelled")
                
                return {
                    "execution_id": execution_id,
                    "cancelled": True,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Pipeline cancellation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/taxonomy/categories")
        async def get_taxonomy_categories():
            """Get all available dataset categories"""
            try:
                # Create MCP task
                task = MCPTask(
                    task_type="taxonomy_info",
                    parameters={}
                )
                
                # Process task
                result = await self.mcp_interface.process_task(task)
                
                if result["status"] == "failed":
                    raise HTTPException(status_code=400, detail=result["error"])
                
                return result["results"]
                
            except Exception as e:
                logger.error(f"Taxonomy retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/taxonomy/categories/{category}")
        async def get_taxonomy_category(category: str):
            """Get specific category information"""
            try:
                # Create MCP task
                task = MCPTask(
                    task_type="taxonomy_info",
                    parameters={"category": category}
                )
                
                # Process task
                result = await self.mcp_interface.process_task(task)
                
                if result["status"] == "failed":
                    raise HTTPException(status_code=400, detail=result["error"])
                
                category_info = result["results"]["category_info"]
                if not category_info:
                    raise HTTPException(status_code=404, detail="Category not found")
                
                return category_info
                
            except Exception as e:
                logger.error(f"Category retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/training/styles")
        async def get_training_styles():
            """Get all available training styles"""
            try:
                # Create MCP task
                task = MCPTask(
                    task_type="style_info",
                    parameters={}
                )
                
                # Process task
                result = await self.mcp_interface.process_task(task)
                
                if result["status"] == "failed":
                    raise HTTPException(status_code=400, detail=result["error"])
                
                return result["results"]
                
            except Exception as e:
                logger.error(f"Training styles retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/training/styles/{style}")
        async def get_training_style(style: str):
            """Get specific training style information"""
            try:
                # Create MCP task
                task = MCPTask(
                    task_type="style_info",
                    parameters={"style": style}
                )
                
                # Process task
                result = await self.mcp_interface.process_task(task)
                
                if result["status"] == "failed":
                    raise HTTPException(status_code=400, detail=result["error"])
                
                style_info = result["results"]["specific_style_info"]
                if not style_info:
                    raise HTTPException(status_code=404, detail="Training style not found")
                
                return {
                    "style": style,
                    "info": style_info,
                    "config_template": result["results"]["specific_config_template"]
                }
                
            except Exception as e:
                logger.error(f"Training style retrieval failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/executions")
        async def list_executions():
            """List all active pipeline executions"""
            try:
                executions = self.orchestrator.list_active_executions()
                
                return {
                    "executions": executions,
                    "total": len(executions),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Executions listing failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _estimate_completion(self, status: Dict[str, Any]) -> Optional[str]:
        """Estimate completion time based on current progress"""
        # This could implement sophisticated estimation logic
        # For now, return a simple estimate
        if status["status"] == "completed":
            return None
        
        progress = status["progress"]
        if progress > 0:
            elapsed_time = (datetime.utcnow() - datetime.fromisoformat(status["start_time"])).total_seconds()
            estimated_total_time = elapsed_time / (progress / 100)
            remaining_time = estimated_total_time - elapsed_time
            return datetime.utcnow().isoformat() + f" (in {remaining_time/3600:.1f} hours)"
        
        return "Unknown"
    
    async def _monitor_execution(self, execution_id: str):
        """Background task to monitor execution"""
        # This would implement continuous monitoring
        # For now, just log the monitoring start
        logger.info(f"Started monitoring execution: {execution_id}")


# Example usage and testing
def test_api_integration():
    """Test API integration"""
    print("Testing API Integration...")
    
    # Create orchestrator and API
    orchestrator = TrainingPipelineOrchestrator()
    api = DatasetTrainingAPI(orchestrator)
    
    # Test health check
    import httpx
    from fastapi.testclient import TestClient
    
    client = TestClient(api.app)
    
    # Test health endpoint
    response = client.get("/health")
    print(f"Health check: {response.status_code}")
    
    # Test taxonomy categories
    response = client.get("/api/taxonomy/categories")
    print(f"Taxonomy categories: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Available categories: {len(data['categories'])}")
    
    # Test training styles
    response = client.get("/api/training/styles")
    print(f"Training styles: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Available styles: {len(data['styles'])}")
    
    print("API integration test completed!")


if __name__ == "__main__":
    test_api_integration()