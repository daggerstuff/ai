"""
Training Style Configuration System for Pixelated Empathy AI
Defines comprehensive training style configurations and management for different therapeutic approaches
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
from datetime import datetime
import json
import uuid
import torch
from pathlib import Path


class TrainingStyle(Enum):
    """Available training styles"""
    FEW_SHOT = "few_shot"
    SELF_SUPERVISED = "self_supervised"
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER_LEARNING = "transfer_learning"
    META_LEARNING = "meta_learning"
    CONTINUAL_LEARNING = "continual_learning"
    DPO = "dpo"  # Direct Preference Optimization


class OptimizationStrategy(Enum):
    """Optimization strategies for training"""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"
    LAMB = "lamb"


class SchedulerType(Enum):
    """Learning rate scheduler types"""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    INVERSE_SQRT = "inverse_sqrt"
    REDUCE_LR_ON_PLATEAU = "reduce_lr_on_plateau"


class LossFunction(Enum):
    """Loss function types"""
    CROSS_ENTROPY = "cross_entropy"
    MEAN_SQUARED_ERROR = "mean_squared_error"
    HUBER = "huber"
    SMOOTH_L1 = "smooth_l1"
    KL_DIVERGENCE = "kl_divergence"
    FOCAL = "focal"
    DPO = "dpo"  # Direct Preference Optimization loss
    RLHF = "rlhf"  # Reinforcement Learning from Human Feedback


class EvaluationMetric(Enum):
    """Evaluation metrics"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    BLEU = "bleu"
    ROUGE = "rouge"
    PERPLEXITY = "perplexity"
    EMOTIONAL_ACCURACY = "emotional_accuracy"
    EMPATHY_SCORE = "empathy_score"
    THERAPEUTIC_APPROPRIATENESS = "therapeutic_appropriateness"


class SafetyThreshold(Enum):
    """Safety thresholds for different training styles"""
    STRICT = "strict"  # Conservative, high safety
    MODERATE = "moderate"  # Balanced safety and performance
    PERMISSIVE = "permissive"  # Performance-focused, lower safety


@dataclass
class BaseTrainingConfig:
    """Base configuration for all training styles"""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    style: TrainingStyle = TrainingStyle.SUPERVISED
    name: str = "default_training"
    description: str = "Default training configuration"

    # Model configuration
    model_name: str = "microsoft/DialoGPT-medium"
    model_type: str = "causal_lm"
    max_sequence_length: int = 512
    tokenizer_name: Optional[str] = None

    # Training parameters
    num_epochs: int = 3
    batch_size: int = 8
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0

    # Optimization
    optimizer: OptimizationStrategy = OptimizationStrategy.ADAMW
    scheduler: SchedulerType = SchedulerType.LINEAR
    loss_function: LossFunction = LossFunction.CROSS_ENTROPY

    # Evaluation
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE, EvaluationMetric.EMPATHY_SCORE
    ])

    # Checkpointing
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True

    # Logging
    logging_steps: int = 10
    logging_dir: str = "./logs"
    report_to: List[str] = field(default_factory=lambda: ["wandb", "tensorboard"])

    # Hardware
    fp16: bool = False
    bf16: bool = True
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True

    # Safety
    safety_threshold: SafetyThreshold = SafetyThreshold.MODERATE
    max_crisis_content_ratio: float = 0.15
    demographic_balance_threshold: float = 0.1
    toxicity_threshold: float = 0.05

    # Additional parameters
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FewShotConfig(BaseTrainingConfig):
    """Configuration for few-shot learning"""
    style: TrainingStyle = TrainingStyle.FEW_SHOT

    # Few-shot specific parameters
    num_shots: int = 5
    num_queries: int = 10
    support_set_size: int = 50
    query_complexity: str = "medium"  # low, medium, high
    shot_selection_strategy: str = "diverse"  # random, diverse, representative

    # Meta-learning parameters
    meta_learning_rate: float = 1e-4
    adaptation_steps: int = 5
    inner_loop_lr: float = 1e-3

    # Evaluation
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE, EvaluationMetric.EMPATHY_SCORE
    ])

    # Safety for few-shot
    safety_threshold: SafetyThreshold = SafetyThreshold.STRICT
    max_crisis_content_ratio: float = 0.1
    demographic_balance_threshold: float = 0.05


@dataclass
class SelfSupervisedConfig(BaseTrainingConfig):
    """Configuration for self-supervised learning"""
    style: TrainingStyle = TrainingStyle.SELF_SUPERVISED

    # Self-supervised specific parameters
    masking_ratio: float = 0.15
    masking_strategy: str = "random"  # random, span, ngram
    contrastive_temperature: float = 0.07
    contrastive_weight: float = 1.0

    # Pre-training objectives
    mlm_probability: float = 0.15
    nsp_probability: float = 0.5
    sop_probability: float = 0.5

    # Contrastive learning
    num_negatives: int = 5
    similarity_metric: str = "cosine"  # cosine, dot_product
    projection_dim: int = 128

    # Data augmentation
    augmentation_enabled: bool = True
    augmentation_strategies: List[str] = field(default_factory=lambda: ["synonym_replacement", "random_deletion"])

    # Evaluation
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.PERPLEXITY, EvaluationMetric.EMPATHY_SCORE
    ])

    # Safety for self-supervised
    safety_threshold: SafetyThreshold = SafetyThreshold.MODERATE
    max_crisis_content_ratio: float = 0.2


@dataclass
class SupervisedConfig(BaseTrainingConfig):
    """Configuration for supervised learning"""
    style: TrainingStyle = TrainingStyle.SUPERVISED

    # Supervised specific parameters
    validation_split: float = 0.2
    test_split: float = 0.1
    stratify_by: Optional[str] = None

    # Label smoothing
    label_smoothing: float = 0.0
    class_weights: Optional[Dict[int, float]] = None

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0

    # Evaluation
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.ACCURACY, EvaluationMetric.PRECISION, EvaluationMetric.RECALL,
        EvaluationMetric.F1_SCORE, EvaluationMetric.EMPATHY_SCORE, EvaluationMetric.THERAPEUTIC_APPROPRIATENESS
    ])

    # Safety for supervised
    safety_threshold: SafetyThreshold = SafetyThreshold.STRICT
    max_crisis_content_ratio: float = 0.15


@dataclass
class UnsupervisedConfig(BaseTrainingConfig):
    """Configuration for unsupervised learning"""
    style: TrainingStyle = TrainingStyle.UNSUPERVISED

    # Unsupervised specific parameters
    num_clusters: int = 10
    cluster_method: str = "kmeans"  # kmeans, dbscan, hierarchical
    dimensionality_reduction: str = "pca"  # pca, t-sne, umap
    reduction_dims: int = 50

    # Clustering parameters
    min_cluster_size: int = 5
    max_cluster_size: int = 1000
    cluster_distance_threshold: float = 0.5

    # Pattern discovery
    pattern_discovery_enabled: bool = True
    pattern_types: List[str] = field(default_factory=lambda: ["sequential", "associative"])

    # Evaluation
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.PERPLEXITY
    ])

    # Safety for unsupervised
    safety_threshold: SafetyThreshold = SafetyThreshold.MODERATE
    max_crisis_content_ratio: float = 0.2


@dataclass
class ReinforcementConfig(BaseTrainingConfig):
    """Configuration for reinforcement learning"""
    style: TrainingStyle = TrainingStyle.REINFORCEMENT

    # Reinforcement specific parameters
    reward_function: str = "therapeutic_outcome"  # therapeutic_outcome, conversation_quality, empathy_score
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01

    # Policy parameters
    policy_type: str = "epsilon_greedy"  # epsilon_greedy, softmax, ucb
    policy_update_frequency: int = 100
    experience_replay_size: int = 10000

    # Q-learning parameters
    q_learning_rate: float = 0.001
    q_discount_factor: float = 0.99
    q_update_frequency: int = 10

    # Environment parameters
    max_episode_length: int = 100
    episode_batch_size: int = 32
    environment_type: str = "therapeutic_conversation"

    # Evaluation
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.ACCURACY, EvaluationMetric.EMPATHY_SCORE, EvaluationMetric.THERAPEUTIC_APPROPRIATENESS
    ])

    # Safety for reinforcement
    safety_threshold: SafetyThreshold = SafetyThreshold.STRICT
    max_crisis_content_ratio: float = 0.1


@dataclass
class TransferLearningConfig(BaseTrainingConfig):
    """Configuration for transfer learning"""
    style: TrainingStyle = TrainingStyle.TRANSFER_LEARNING

    # Transfer learning specific parameters
    source_model: str = "microsoft/DialoGPT-medium"
    target_domain: str = "therapeutic_conversations"
    transfer_method: str = "fine_tuning"  # fine_tuning, feature_extraction, adapter_tuning

    # Fine-tuning parameters
    layers_to_freeze: int = 0
    layer_selection_strategy: str = "bottom_up"  # bottom_up, top_down, selective
    fine_tuning_lr_multiplier: float = 0.1

    # Domain adaptation
    domain_adaptation_enabled: bool = True
    domain_adaptation_weight: float = 0.5
    adversarial_training: bool = False

    # Knowledge distillation
    knowledge_distillation_enabled: bool = False
    temperature: float = 3.0
    alpha: float = 0.5

    # Evaluation
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE, EvaluationMetric.EMPATHY_SCORE
    ])

    # Safety for transfer learning
    safety_threshold: SafetyThreshold = SafetyThreshold.MODERATE
    max_crisis_content_ratio: float = 0.15


@dataclass
class MetaLearningConfig(BaseTrainingConfig):
    """Configuration for meta learning"""
    style: TrainingStyle = TrainingStyle.META_LEARNING

    # Meta learning specific parameters
    meta_learning_rate: float = 0.001
    inner_loop_lr: float = 0.01
    adaptation_steps: int = 5
    meta_batch_size: int = 4

    # Task distribution
    num_tasks: int = 100
    task_distribution: str = "uniform"  # uniform, gaussian, mixture
    task_complexity: str = "medium"  # low, medium, high

    # MAML parameters
    first_order_maml: bool = True
    maml_lr: float = 0.01
    maml_meta_lr: float = 0.001

    # Prototypical networks
    prototypical_networks: bool = False
    prototype_dim: int = 64
    distance_metric: str = "euclidean"  # euclidean, cosine, manhattan

    # Evaluation
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE
    ])

    # Safety for meta learning
    safety_threshold: SafetyThreshold = SafetyThreshold.MODERATE
    max_crisis_content_ratio: float = 0.15


@dataclass
class ContinualLearningConfig(BaseTrainingConfig):
    """Configuration for continual learning"""
    style: TrainingStyle = TrainingStyle.CONTINUAL_LEARNING

    # Continual learning specific parameters
    memory_buffer_size: int = 1000
    memory_selection_strategy: str = "herding"  # herding, random, uncertainty
    regularization_strength: float = 0.1
    forgetting_rate: float = 0.01

    # Experience replay
    experience_replay_enabled: bool = True
    replay_frequency: int = 100
    replay_sample_size: int = 32

    # Regularization methods
    ewc_lambda: float = 0.1  # Elastic Weight Consolidation
    lwf_alpha: float = 0.5  # Learning without Forgetting
    mas_lambda: float = 0.1  # Memory Aware Synapses

    # Task boundaries
    task_boundary_detection: str = "automatic"  # automatic, manual, hybrid
    task_similarity_threshold: float = 0.8

    # Evaluation
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE
    ])

    # Safety for continual learning
    safety_threshold: SafetyThreshold = SafetyThreshold.STRICT
    max_crisis_content_ratio: float = 0.12


@dataclass
class DPOConfig(BaseTrainingConfig):
    """Configuration for Direct Preference Optimization (DPO) training.

    DPO is a preference learning method that directly optimizes language models
    to align with human preferences without requiring explicit reward modeling.

    Based on the paper: "Direct Preference Optimization: Your Language Model is
    Secretly a Reward Model" (Rafailov et al., 2023)
    """
    style: TrainingStyle = TrainingStyle.DPO
    loss_function: LossFunction = LossFunction.DPO

    # DPO-specific parameters
    beta: float = 0.1  # Temperature parameter controlling deviation from reference
    reference_free: bool = False  # Use reference-free DPO variant
    label_smoothing: float = 0.0  # Smoothing for preference labels

    # Preference data format
    preference_data_format: str = "chosen_rejected"  # chosen_rejected, ranked, pairwise
    chosen_column: str = "chosen"  # Column name for preferred response
    rejected_column: str = "rejected"  # Column name for rejected response
    prompt_column: str = "prompt"  # Column name for prompt/context

    # Reference model
    reference_model: Optional[str] = None  # Path to reference model (if different from base)
    ref_model_mixins: bool = False  # Use mixin for reference model
    sync_ref_model: bool = False  # Sync reference model with policy

    # Loss variants
    loss_type: str = "sigmoid"  # sigmoid, hinge, ipo, kto, sppo
    label_pad_token_id: int = -100
    padding_value: int = 0

    # Training parameters
    max_prompt_length: int = 512
    max_completion_length: int = 512
    truncation_mode: str = "keep_end"  # keep_start, keep_end

    # Implicit reward model
    generate_during_eval: bool = False
    precompute_ref_log_probs: bool = False

    # Regularization
    rpo_alpha: Optional[float] = None  # Relative policy optimization weight
    sft_loss_weight: float = 0.0  # SFT auxiliary loss weight

    # Dataset processing
    dataset_num_proc: int = 4
    remove_unused_columns: bool = True

    # Safety for DPO
    safety_threshold: SafetyThreshold = SafetyThreshold.STRICT
    max_crisis_content_ratio: float = 0.05  # Very low for preference learning
    toxicity_threshold: float = 0.02  # Stricter toxicity threshold for alignment

    # Evaluation
    evaluation_metrics: List[EvaluationMetric] = field(default_factory=lambda: [
        EvaluationMetric.ACCURACY, EvaluationMetric.F1_SCORE,
        EvaluationMetric.EMPATHY_SCORE, EvaluationMetric.THERAPEUTIC_APPROPRIATENESS
    ])

    # DPO-specific metadata
    metadata: Dict[str, Any] = field(default_factory=lambda: {
        "training_type": "preference_learning",
        "alignment_method": "direct_preference_optimization",
        "suitable_for": ["response_quality", "safety_alignment", "therapeutic_style"]
    })


class TrainingStyleManager:
    """Manages training style configurations and selection"""

    def __init__(self):
        self.config_registry = self._build_config_registry()
        self.style_requirements = self._build_style_requirements()
        self.optimization_strategies = self._build_optimization_strategies()

    def _build_config_registry(self) -> Dict[TrainingStyle, type]:
        """Build registry of configuration classes"""
        return {
            TrainingStyle.FEW_SHOT: FewShotConfig,
            TrainingStyle.SELF_SUPERVISED: SelfSupervisedConfig,
            TrainingStyle.SUPERVISED: SupervisedConfig,
            TrainingStyle.UNSUPERVISED: UnsupervisedConfig,
            TrainingStyle.REINFORCEMENT: ReinforcementConfig,
            TrainingStyle.TRANSFER_LEARNING: TransferLearningConfig,
            TrainingStyle.META_LEARNING: MetaLearningConfig,
            TrainingStyle.CONTINUAL_LEARNING: ContinualLearningConfig,
            TrainingStyle.DPO: DPOConfig,
        }

    def _build_style_requirements(self) -> Dict[TrainingStyle, Dict[str, Any]]:
        """Build requirements for each training style"""
        return {
            TrainingStyle.FEW_SHOT: {
                "min_dataset_size": 50,
                "max_dataset_size": 10000,
                "requires_labels": True,
                "requires_high_quality": True,
                "suitable_domains": ["therapeutic", "clinical", "conversational"],
                "computational_requirements": "medium",
                "memory_requirements": "medium"
            },
            TrainingStyle.SELF_SUPERVISED: {
                "min_dataset_size": 1000,
                "max_dataset_size": None,
                "requires_labels": False,
                "requires_high_quality": False,
                "suitable_domains": ["conversational", "synthetic", "multimodal"],
                "computational_requirements": "high",
                "memory_requirements": "high"
            },
            TrainingStyle.SUPERVISED: {
                "min_dataset_size": 100,
                "max_dataset_size": None,
                "requires_labels": True,
                "requires_high_quality": True,
                "suitable_domains": ["therapeutic", "clinical", "conversational"],
                "computational_requirements": "medium",
                "memory_requirements": "medium"
            },
            TrainingStyle.UNSUPERVISED: {
                "min_dataset_size": 500,
                "max_dataset_size": None,
                "requires_labels": False,
                "requires_high_quality": False,
                "suitable_domains": ["conversational", "synthetic"],
                "computational_requirements": "medium",
                "memory_requirements": "medium"
            },
            TrainingStyle.REINFORCEMENT: {
                "min_dataset_size": 200,
                "max_dataset_size": 50000,
                "requires_labels": True,
                "requires_high_quality": True,
                "suitable_domains": ["therapeutic", "clinical"],
                "computational_requirements": "very_high",
                "memory_requirements": "high"
            },
            TrainingStyle.TRANSFER_LEARNING: {
                "min_dataset_size": 100,
                "max_dataset_size": 50000,
                "requires_labels": True,
                "requires_high_quality": True,
                "suitable_domains": ["therapeutic", "clinical", "conversational"],
                "computational_requirements": "medium",
                "memory_requirements": "medium"
            },
            TrainingStyle.META_LEARNING: {
                "min_dataset_size": 1000,
                "max_dataset_size": 100000,
                "requires_labels": True,
                "requires_high_quality": True,
                "suitable_domains": ["therapeutic", "conversational"],
                "computational_requirements": "very_high",
                "memory_requirements": "very_high"
            },
            TrainingStyle.CONTINUAL_LEARNING: {
                "min_dataset_size": 500,
                "max_dataset_size": None,
                "requires_labels": True,
                "requires_high_quality": True,
                "suitable_domains": ["therapeutic", "clinical", "conversational"],
                "computational_requirements": "high",
                "memory_requirements": "very_high"
            },
            TrainingStyle.DPO: {
                "min_dataset_size": 500,
                "max_dataset_size": 100000,
                "requires_labels": True,  # Requires preference pairs (chosen/rejected)
                "requires_high_quality": True,
                "requires_preference_pairs": True,
                "suitable_domains": ["therapeutic", "clinical", "conversational", "safety_alignment"],
                "computational_requirements": "high",
                "memory_requirements": "high",
                "description": "Direct Preference Optimization for aligning model responses with human preferences"
            }
        }

    def _build_optimization_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Build optimization strategies for different scenarios"""
        return {
            "fast_training": {
                "num_epochs": 1,
                "batch_size": 16,
                "learning_rate": 5e-5,
                "logging_steps": 50,
                "evaluation_strategy": "no"
            },
            "balanced": {
                "num_epochs": 3,
                "batch_size": 8,
                "learning_rate": 2e-5,
                "logging_steps": 10,
                "evaluation_strategy": "steps"
            },
            "high_quality": {
                "num_epochs": 5,
                "batch_size": 4,
                "learning_rate": 1e-5,
                "logging_steps": 5,
                "evaluation_strategy": "steps"
            },
            "resource_efficient": {
                "num_epochs": 2,
                "batch_size": 32,
                "learning_rate": 3e-5,
                "gradient_accumulation_steps": 4,
                "fp16": True
            }
        }

    def create_config(self, style: TrainingStyle, **kwargs) -> BaseTrainingConfig:
        """Create a training configuration for a specific style"""
        config_class = self.config_registry.get(style)
        if not config_class:
            raise ValueError(f"Unknown training style: {style}")

        return config_class(**kwargs)

    def select_optimal_style(self, dataset_metadata: Dict[str, Any],
                           training_goals: Dict[str, Any]) -> TrainingStyle:
        """Select the optimal training style based on dataset and goals"""
        # Score each style
        style_scores = {}

        for style in TrainingStyle:
            score = self._score_style_suitability(style, dataset_metadata, training_goals)
            style_scores[style] = score

        # Return the best style
        best_style = max(style_scores, key=style_scores.get)
        return best_style

    def _score_style_suitability(self, style: TrainingStyle,
                               dataset_metadata: Dict[str, Any],
                               training_goals: Dict[str, Any]) -> float:
        """Score how suitable a training style is for given dataset and goals"""
        requirements = self.style_requirements.get(style, {})
        score = 0.0

        # Check dataset size
        dataset_size = dataset_metadata.get("record_count", 0)
        min_size = requirements.get("min_dataset_size", 0)
        max_size = requirements.get("max_dataset_size", float('inf'))

        if dataset_size >= min_size and dataset_size <= max_size:
            score += 0.3
        else:
            return 0.0  # Unsuitable due to size constraints

        # Check label requirements
        has_labels = dataset_metadata.get("has_labels", False)
        requires_labels = requirements.get("requires_labels", False)

        if has_labels == requires_labels:
            score += 0.2
        elif requires_labels and not has_labels:
            return 0.0  # Unsuitable due to label requirements

        # Check quality requirements
        quality_score = dataset_metadata.get("quality_score", 0.0)
        requires_high_quality = requirements.get("requires_high_quality", False)

        if requires_high_quality and quality_score >= 0.8:
            score += 0.2
        elif not requires_high_quality:
            score += 0.2
        elif requires_high_quality and quality_score < 0.8:
            score += 0.1  # Partial credit

        # Check domain suitability
        dataset_domain = dataset_metadata.get("category", "")
        suitable_domains = requirements.get("suitable_domains", [])

        if dataset_domain in suitable_domains:
            score += 0.2
        elif suitable_domains:
            score += 0.1  # Partial credit for related domains

        # Check computational requirements vs available resources
        comp_requirement = requirements.get("computational_requirements", "medium")
        available_compute = training_goals.get("available_compute", "medium")

        comp_levels = {"low": 1, "medium": 2, "high": 3, "very_high": 4}
        required_level = comp_levels.get(comp_requirement, 2)
        available_level = comp_levels.get(available_compute, 2)

        if available_level >= required_level:
            score += 0.1
        else:
            score -= 0.1  # Penalty for insufficient compute

        return max(score, 0.0)

    def optimize_config(self, config: BaseTrainingConfig,
                       optimization_strategy: str = "balanced") -> BaseTrainingConfig:
        """Optimize configuration based on strategy"""
        strategy_params = self.optimization_strategies.get(optimization_strategy, {})

        # Apply optimization parameters
        for key, value in strategy_params.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    def validate_config(self, config: BaseTrainingConfig) -> List[str]:
        """Validate training configuration"""
        errors = []

        # Basic validation
        if config.num_epochs <= 0:
            errors.append("Number of epochs must be positive")

        if config.batch_size <= 0:
            errors.append("Batch size must be positive")

        if config.learning_rate <= 0:
            errors.append("Learning rate must be positive")

        # Style-specific validation
        if isinstance(config, FewShotConfig):
            if config.num_shots <= 0:
                errors.append("Number of shots must be positive")
            if config.num_queries <= 0:
                errors.append("Number of queries must be positive")

        elif isinstance(config, SelfSupervisedConfig):
            if config.masking_ratio < 0 or config.masking_ratio > 1:
                errors.append("Masking ratio must be between 0 and 1")
            if config.contrastive_temperature <= 0:
                errors.append("Contrastive temperature must be positive")

        elif isinstance(config, SupervisedConfig):
            if config.validation_split < 0 or config.validation_split > 1:
                errors.append("Validation split must be between 0 and 1")
            if config.test_split < 0 or config.test_split > 1:
                errors.append("Test split must be between 0 and 1")

        elif isinstance(config, ReinforcementConfig):
            if config.exploration_rate < 0 or config.exploration_rate > 1:
                errors.append("Exploration rate must be between 0 and 1")
            if config.q_discount_factor < 0 or config.q_discount_factor > 1:
                errors.append("Q-learning discount factor must be between 0 and 1")

        elif isinstance(config, TransferLearningConfig):
            if config.layers_to_freeze < 0:
                errors.append("Layers to freeze must be non-negative")
            if config.fine_tuning_lr_multiplier <= 0:
                errors.append("Fine-tuning learning rate multiplier must be positive")

        elif isinstance(config, MetaLearningConfig):
            if config.num_tasks <= 0:
                errors.append("Number of tasks must be positive")
            if config.adaptation_steps <= 0:
                errors.append("Adaptation steps must be positive")

        elif isinstance(config, ContinualLearningConfig):
            if config.memory_buffer_size < 0:
                errors.append("Memory buffer size must be non-negative")
            if config.regularization_strength < 0:
                errors.append("Regularization strength must be non-negative")

        # Safety validation
        if config.safety_threshold == SafetyThreshold.STRICT:
            if config.max_crisis_content_ratio > 0.15:
                errors.append("Strict safety threshold requires crisis content ratio <= 0.15")
            if config.toxicity_threshold > 0.05:
                errors.append("Strict safety threshold requires toxicity threshold <= 0.05")

        return errors

    def get_config_template(self, style: TrainingStyle) -> Dict[str, Any]:
        """Get a template configuration for a specific style"""
        config_class = self.config_registry.get(style)
        if not config_class:
            raise ValueError(f"Unknown training style: {style}")

        # Create default instance
        config = config_class()

        # Convert to dict with helpful defaults
        template = config.__dict__.copy()

        # Add style-specific guidance
        guidance = self._get_style_guidance(style)
        template["_guidance"] = guidance

        return template

    def _get_style_guidance(self, style: TrainingStyle) -> Dict[str, str]:
        """Get guidance for a specific training style"""
        guidance = {
            TrainingStyle.FEW_SHOT: {
                "description": "Few-shot learning for limited data scenarios",
                "best_for": "Small, high-quality datasets with clear patterns",
                "requirements": "High-quality labeled data, clear task definition",
                "pitfalls": "Overfitting on small datasets, poor generalization"
            },
            TrainingStyle.SELF_SUPERVISED: {
                "description": "Self-supervised learning from unlabeled data",
                "best_for": "Large datasets without labels, representation learning",
                "requirements": "Large amounts of text data, computational resources",
                "pitfalls": "Requires significant compute, may not capture specific patterns"
            },
            TrainingStyle.SUPERVISED: {
                "description": "Standard supervised learning with labeled data",
                "best_for": "Well-labeled datasets with clear objectives",
                "requirements": "High-quality labeled data, balanced classes",
                "pitfalls": "Requires labeled data, may overfit on small datasets"
            },
            TrainingStyle.UNSUPERVISED: {
                "description": "Unsupervised learning for pattern discovery",
                "best_for": "Exploratory analysis, pattern discovery, clustering",
                "requirements": "Sufficient data for meaningful patterns",
                "pitfalls": "Results may be hard to interpret, requires domain expertise"
            },
            TrainingStyle.REINFORCEMENT: {
                "description": "Reinforcement learning for strategy optimization",
                "best_for": "Optimizing conversation strategies, policy learning",
                "requirements": "Clear reward signals, environment simulation",
                "pitfalls": "Complex setup, requires careful reward design"
            },
            TrainingStyle.TRANSFER_LEARNING: {
                "description": "Transfer learning from pre-trained models",
                "best_for": "Adapting general models to therapeutic domains",
                "requirements": "Pre-trained model, domain-specific data",
                "pitfalls": "Domain mismatch, catastrophic forgetting"
            },
            TrainingStyle.META_LEARNING: {
                "description": "Meta learning for fast adaptation",
                "best_for": "Learning to learn, few-shot adaptation",
                "requirements": "Diverse tasks, sufficient meta-training data",
                "pitfalls": "Complex implementation, requires careful task design"
            },
            TrainingStyle.CONTINUAL_LEARNING: {
                "description": "Continual learning for evolving data",
                "best_for": "Streaming data, evolving conversation patterns",
                "requirements": "Memory management, regularization strategies",
                "pitfalls": "Forgetting previous knowledge, memory constraints"
            }
        }

        return guidance.get(style, {})


# Example usage and testing
def test_training_styles():
    """Test the training styles system"""
    print("Testing Training Styles System...")

    manager = TrainingStyleManager()

    # Test config creation
    config = manager.create_config(TrainingStyle.FEW_SHOT, num_shots=10)
    print(f"Created few-shot config with {config.num_shots} shots")

    # Test style selection
    dataset_metadata = {
        "record_count": 500,
        "has_labels": True,
        "quality_score": 0.9,
        "category": "therapeutic",
        "available_compute": "high"
    }

    training_goals = {
        "objective": "high_accuracy",
        "available_compute": "high",
        "time_constraints": "moderate"
    }

    optimal_style = manager.select_optimal_style(dataset_metadata, training_goals)
    print(f"Optimal training style: {optimal_style}")

    # Test config validation
    errors = manager.validate_config(config)
    print(f"Config validation errors: {errors}")

    # Test config optimization
    optimized_config = manager.optimize_config(config, "high_quality")
    print(f"Optimized learning rate: {optimized_config.learning_rate}")
    print(f"Optimized epochs: {optimized_config.num_epochs}")

    # Test config template
    template = manager.get_config_template(TrainingStyle.SUPERVISED)
    print(f"Supervised config template keys: {list(template.keys())}")

    print("Training styles test completed!")


if __name__ == "__main__":
    test_training_styles()
