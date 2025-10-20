"""
Pixel Training System Components

Multi-objective training, psychology integration, and comprehensive
monitoring systems.
"""

# Core training components
from .data_augmentation import (
    AugmentationConfig,
    ContextExpander,
    CrisisScenarioGenerator,
    DataAugmentationPipeline,
    DialogueVariationGenerator,
)
from .data_loader import (
    DataLoaderConfig,
    PixelDataLoader,
    TherapeuticConversationDataset,
    setup_data_loaders,
)
from .training_config import (
    ComputeConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
    TrainingConfigManager,
    create_training_config,
)

# Future components
# from .multi_objective_loss import MultiObjectiveLoss
# from .dynamic_loss_scheduler import DynamicLossScheduler
# from .psychology_trainer import PsychologyTrainer

__all__ = [
    # Data loading
    "PixelDataLoader",
    "DataLoaderConfig",
    "TherapeuticConversationDataset",
    "setup_data_loaders",
    # Configuration
    "TrainingConfigManager",
    "TrainingConfig",
    "ComputeConfig",
    "ModelConfig",
    "OutputConfig",
    "create_training_config",
    # Future
    # "MultiObjectiveLoss",
    # "DynamicLossScheduler",
    # "PsychologyTrainer"
]

__version__ = "0.1.0"
