"""
Sampling utilities for creating balanced train/validation/test splits.
Implements stratified sampling and various balancing techniques for mental health datasets.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
from enum import Enum
import random
import math
from collections import Counter
import logging
from .conversation_schema import Conversation
from .label_taxonomy import LabelBundle, TherapeuticResponseType, CrisisLevelType

logger = logging.getLogger(__name__)


class SplitType(Enum):
    """Types of dataset splits"""
    RANDOM = "random"
    STRATIFIED = "stratified"
    TEMPORAL = "temporal"
    DEMOGRAPHIC = "demographic"
    THERAPEUTIC_MODALITY = "therapeutic_modality"


@dataclass
class SplitConfig:
    """Configuration for dataset splitting"""
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    split_type: SplitType = SplitType.STRATIFIED
    stratify_by: Optional[str] = "crisis_level"  # Field to stratify by
    random_seed: Optional[int] = 42
    ensure_minimum_samples: int = 5  # Minimum samples per class
    max_imbalance_ratio: float = 0.2  # Maximum allowed imbalance ratio
    include_rare_classes: bool = True  # Whether to include rare classes


@dataclass
class DatasetSplit:
    """Represents a train/validation/test split"""
    train: List[Tuple[Conversation, LabelBundle]]
    validation: List[Tuple[Conversation, LabelBundle]]
    test: List[Tuple[Conversation, LabelBundle]]
    split_stats: Dict[str, Any]


class Sampler:
    """Main class for sampling and splitting datasets"""
    
    def __init__(self, config: Optional[SplitConfig] = None):
        self.config = config or SplitConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)
        self.logger = logging.getLogger(__name__)

    def split_dataset(self, 
                     conversations: List[Conversation], 
                     label_bundles: List[LabelBundle]) -> DatasetSplit:
        """Split dataset according to configuration"""
        if len(conversations) != len(label_bundles):
            raise ValueError("Conversations and label bundles must have the same length")
        
        # Pair conversations with their labels
        paired_data = list(zip(conversations, label_bundles))
        
        # Apply the appropriate splitting strategy
        if self.config.split_type == SplitType.RANDOM:
            splits = self._random_split(paired_data)
        elif self.config.split_type == SplitType.STRATIFIED:
            splits = self._stratified_split(paired_data)
        elif self.config.split_type == SplitType.TEMPORAL:
            splits = self._temporal_split(paired_data)
        elif self.config.split_type == SplitType.DEMOGRAPHIC:
            splits = self._demographic_split(paired_data)
        elif self.config.split_type == SplitType.THERAPEUTIC_MODALITY:
            splits = self._modality_split(paired_data)
        else:
            raise ValueError(f"Unsupported split type: {self.config.split_type}")
        
        # Calculate and include statistics
        split_stats = self._calculate_split_statistics(paired_data, splits)
        
        return DatasetSplit(
            train=splits[0],
            validation=splits[1], 
            test=splits[2],
            split_stats=split_stats
        )

    def _random_split(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> Tuple[List, List, List]:
        """Perform random split of the data"""
        # Shuffle the data
        shuffled_data = paired_data.copy()
        random.shuffle(shuffled_data)
        
        # Calculate split indices
        total = len(shuffled_data)
        train_end = int(total * self.config.train_ratio)
        val_end = train_end + int(total * self.config.validation_ratio)
        
        # Split the data
        train = shuffled_data[:train_end]
        validation = shuffled_data[train_end:val_end]
        test = shuffled_data[val_end:]
        
        return train, validation, test

    def _stratified_split(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> Tuple[List, List, List]:
        """Perform stratified split to maintain label distributions"""
        # Group data by the stratification field
        if self.config.stratify_by == "crisis_level":
            strata = self._group_by_crisis_level(paired_data)
        elif self.config.stratify_by == "therapeutic_response":
            strata = self._group_by_therapeutic_response(paired_data)
        elif self.config.stratify_by == "therapy_modality":
            strata = self._group_by_therapy_modality(paired_data)
        else:
            # Default to crisis level
            strata = self._group_by_crisis_level(paired_data)
        
        # Split each stratum separately
        train, validation, test = [], [], []
        
        for stratum_label, stratum_data in strata.items():
            if len(stratum_data) < 3:  # Need minimum samples for all splits
                # If stratum is too small, randomly assign to splits
                for item in stratum_data:
                    rand_val = random.random()
                    if rand_val < self.config.train_ratio:
                        train.append(item)
                    elif rand_val < self.config.train_ratio + self.config.validation_ratio:
                        validation.append(item)
                    else:
                        test.append(item)
            else:
                # Calculate split sizes for this stratum
                stratum_size = len(stratum_data)
                train_size = max(1, int(stratum_size * self.config.train_ratio))
                val_size = max(1, int(stratum_size * self.config.validation_ratio))
                
                # Shuffle and split this stratum
                shuffled_stratum = stratum_data.copy()
                random.shuffle(shuffled_stratum)
                
                train.extend(shuffled_stratum[:train_size])
                validation.extend(shuffled_stratum[train_size:train_size + val_size])
                test.extend(shuffled_stratum[train_size + val_size:])
        
        return train, validation, test

    def _temporal_split(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> Tuple[List, List, List]:
        """Perform temporal split based on conversation timestamps"""
        # Sort by conversation creation time
        sorted_data = sorted(
            paired_data, 
            key=lambda x: x[0].created_at  # Sort by conversation creation time
        )
        
        # Calculate split indices
        total = len(sorted_data)
        train_end = int(total * self.config.train_ratio)
        val_end = train_end + int(total * self.config.validation_ratio)
        
        # Split the data temporally
        train = sorted_data[:train_end]
        validation = sorted_data[train_end:val_end]
        test = sorted_data[val_end:]
        
        return train, validation, test

    def _group_by_crisis_level(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> Dict[str, List]:
        """Group data by crisis level"""
        strata = {}
        
        for conv, bundle in paired_data:
            crisis_level = "none"
            if bundle.crisis_label:
                crisis_level = bundle.crisis_label.crisis_level.value
            else:
                crisis_level = "no_crisis"
            
            if crisis_level not in strata:
                strata[crisis_level] = []
            strata[crisis_level].append((conv, bundle))
        
        return strata

    def _group_by_therapeutic_response(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> Dict[str, List]:
        """Group data by therapeutic response types"""
        strata = {}
        
        for conv, bundle in paired_data:
            # Group by the most common therapeutic response type in the conversation
            if bundle.therapeutic_response_labels:
                # Get the most frequent response type
                response_types = [label.response_type.value for label in bundle.therapeutic_response_labels]
                counter = Counter(response_types)
                most_common = counter.most_common(1)[0][0]
                
                if most_common not in strata:
                    strata[most_common] = []
                strata[most_common].append((conv, bundle))
            else:
                if "no_response_labels" not in strata:
                    strata["no_response_labels"] = []
                strata["no_response_labels"].append((conv, bundle))
        
        return strata

    def _group_by_therapy_modality(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> Dict[str, List]:
        """Group data by therapy modality"""
        strata = {}
        
        for conv, bundle in paired_data:
            modality = "unknown"
            if bundle.therapy_modality_label:
                modality = bundle.therapy_modality_label.modality.value
            else:
                modality = "no_modality_label"
            
            if modality not in strata:
                strata[modality] = []
            strata[modality].append((conv, bundle))
        
        return strata

    def _demographic_split(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> Tuple[List, List, List]:
        """Split based on demographic information"""
        # This would be implemented based on demographic labels
        # For now, fallback to stratified split by crisis level
        return self._stratified_split(paired_data)

    def _modality_split(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> Tuple[List, List, List]:
        """Split based on therapeutic modality"""
        # This would be implemented based on modality labels
        # For now, fallback to stratified split by therapy modality
        return self._stratified_split(paired_data)

    def _calculate_split_statistics(self, all_data: List[Tuple[Conversation, LabelBundle]], 
                                  splits: Tuple[List, List, List]) -> Dict[str, Any]:
        """Calculate statistics about the dataset splits"""
        train, validation, test = splits
        
        stats = {
            "total_samples": len(all_data),
            "split_ratios": {
                "train": len(train) / len(all_data) if all_data else 0,
                "validation": len(validation) / len(all_data) if all_data else 0,
                "test": len(test) / len(all_data) if all_data else 0
            },
            "split_counts": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test)
            },
            "label_distributions": {
                "train": self._calculate_label_distribution(train),
                "validation": self._calculate_label_distribution(validation),
                "test": self._calculate_label_distribution(test)
            },
            "imbalance_metrics": self._calculate_imbalance_metrics(splits)
        }
        
        return stats

    def _calculate_label_distribution(self, split_data: List[Tuple[Conversation, LabelBundle]]) -> Dict[str, Any]:
        """Calculate label distribution within a split"""
        crisis_counts = Counter()
        response_counts = Counter()
        total_conversations = len(split_data)
        
        for conv, bundle in split_data:
            # Count crisis levels
            if bundle.crisis_label:
                crisis_counts[bundle.crisis_label.crisis_level.value] += 1
            else:
                crisis_counts["no_crisis"] += 1
            
            # Count therapeutic response types
            for label in bundle.therapeutic_response_labels:
                response_counts[label.response_type.value] += 1
        
        return {
            "crisis_distribution": {k: v/total_conversations if total_conversations > 0 else 0 
                                   for k, v in crisis_counts.items()},
            "response_distribution": {k: v/total_conversations if total_conversations > 0 else 0 
                                     for k, v in response_counts.items()},
            "total_conversations": total_conversations
        }

    def _calculate_imbalance_metrics(self, splits: Tuple[List, List, List]) -> Dict[str, float]:
        """Calculate metrics related to class imbalance across splits"""
        train, validation, test = splits
        
        # Calculate imbalance for crisis levels
        train_dist = self._calculate_label_distribution(train)["crisis_distribution"]
        val_dist = self._calculate_label_distribution(validation)["crisis_distribution"]
        test_dist = self._calculate_label_distribution(test)["crisis_distribution"]
        
        # Calculate differences in distribution between splits
        max_crisis_diff = 0.0
        all_crisis_levels = set(train_dist.keys()).union(val_dist.keys()).union(test_dist.keys())
        
        for level in all_crisis_levels:
            train_ratio = train_dist.get(level, 0)
            val_ratio = val_dist.get(level, 0)
            test_ratio = test_dist.get(level, 0)
            
            max_diff = max(
                abs(train_ratio - val_ratio),
                abs(train_ratio - test_ratio),
                abs(val_ratio - test_ratio)
            )
            max_crisis_diff = max(max_crisis_diff, max_diff)
        
        return {
            "max_crisis_level_difference": max_crisis_diff
        }

    def balance_dataset(self, 
                       conversations: List[Conversation],
                       label_bundles: List[LabelBundle],
                       balancing_strategy: str = "oversample") -> Tuple[List[Conversation], List[LabelBundle]]:
        """
        Balance the dataset using various strategies
        Strategies: 'oversample', 'undersample', 'smote' (simplified)
        """
        paired_data = list(zip(conversations, label_bundles))
        
        if balancing_strategy == "oversample":
            balanced_data = self._oversample(paired_data)
        elif balancing_strategy == "undersample":
            balanced_data = self._undersample(paired_data)
        elif balancing_strategy == "smote":
            balanced_data = self._smote_like_oversample(paired_data)
        else:
            raise ValueError(f"Unknown balancing strategy: {balancing_strategy}")
        
        # Separate back into conversations and labels
        balanced_convs, balanced_labels = zip(*balanced_data) if balanced_data else ([], [])
        
        return list(balanced_convs), list(balanced_labels)

    def _oversample(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> List[Tuple[Conversation, LabelBundle]]:
        """Oversample minority classes to balance the dataset"""
        # Group by crisis level for balancing
        strata = self._group_by_crisis_level(paired_data)
        
        if not strata:
            return paired_data  # Nothing to balance
        
        # Find the largest stratum size
        max_size = max(len(data) for data in strata.values())
        
        # Oversample smaller strata to match max size
        balanced_data = []
        for stratum_label, stratum_data in strata.items():
            if len(stratum_data) < max_size:
                # Calculate how many times to repeat the stratum
                repeats = math.ceil(max_size / len(stratum_data))
                
                # Repeat the stratum data with replacement
                for _ in range(repeats):
                    # Add shuffled copies to introduce variation
                    shuffled_copy = stratum_data.copy()
                    random.shuffle(shuffled_copy)
                    for item in shuffled_copy:
                        if len(balanced_data) >= max_size:
                            break
                        balanced_data.append(item)
                # Limit to exactly max_size for this stratum
                stratum_with_repeats = stratum_data * repeats
                balanced_data.extend(stratum_with_repeats[:max_size - len([x for x in balanced_data if self._get_crisis_level(x[1]) == stratum_label])])
            else:
                balanced_data.extend(stratum_data)
        
        return balanced_data

    def _undersample(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> List[Tuple[Conversation, LabelBundle]]:
        """Undersample majority classes to balance the dataset"""
        strata = self._group_by_crisis_level(paired_data)
        
        if not strata:
            return paired_data
        
        # Find the smallest stratum size
        min_size = min(len(data) for data in strata.values()) if strata.values() else 0
        
        # Undersample all strata to match min size
        balanced_data = []
        for stratum_data in strata.values():
            if len(stratum_data) > min_size:
                # Randomly select min_size samples from this stratum
                selected = random.sample(stratum_data, min_size)
                balanced_data.extend(selected)
            else:
                balanced_data.extend(stratum_data)
        
        return balanced_data

    def _smote_like_oversample(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> List[Tuple[Conversation, LabelBundle]]:
        """Simplified SMOTE-like oversampling using data augmentation"""
        from .data_augmentation import create_default_augmenter, DataAugmenter
        
        # For now, we'll use a simple approach: augment minority class samples
        strata = self._group_by_crisis_level(paired_data)
        
        if not strata:
            return paired_data
        
        # Find the largest stratum size
        max_size = max(len(data) for data in strata.values()) if strata.values() else 0
        balanced_data = []
        
        # Initialize augmenter for generating synthetic samples
        augmenter = create_default_augmenter()
        
        for stratum_label, stratum_data in strata.items():
            if len(stratum_data) < max_size:
                # Add original samples
                balanced_data.extend(stratum_data)
                
                # Generate synthetic samples
                needed = max_size - len(stratum_data)
                synthetic_count = 0
                
                while synthetic_count < needed and stratum_data:
                    # Pick a random sample from this stratum
                    original_sample = random.choice(stratum_data)
                    orig_conv, orig_labels = original_sample
                    
                    # Augment the conversation
                    augmented_conv = augmenter.augment_conversation(orig_conv)
                    
                    # Create corresponding augmented labels
                    augmented_labels = augmenter.augment_label_bundle(orig_labels, augmented_conv)
                    
                    balanced_data.append((augmented_conv, augmented_labels))
                    synthetic_count += 1
            else:
                balanced_data.extend(stratum_data)
        
        return balanced_data

    def _get_crisis_level(self, bundle: LabelBundle) -> str:
        """Helper to get crisis level for a bundle"""
        if bundle.crisis_label:
            return bundle.crisis_label.crisis_level.value
        return "no_crisis"


class AdvancedSampler:
    """Advanced sampling techniques with more sophisticated algorithms"""
    
    def __init__(self, config: Optional[SplitConfig] = None):
        self.config = config or SplitConfig()
        if self.config.random_seed is not None:
            random.seed(self.config.random_seed)

    def cluster_aware_split(self, 
                           conversations: List[Conversation],
                           label_bundles: List[LabelBundle],
                           cluster_labels: List[str]) -> DatasetSplit:
        """
        Split while ensuring that items from the same cluster don't appear in multiple splits.
        This is important for preventing data leakage when conversations are clustered by client.
        """
        if len(conversations) != len(label_bundles) or len(conversations) != len(cluster_labels):
            raise ValueError("Conversations, labels, and cluster labels must have the same length")
        
        # Group by clusters
        clusters = {}
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in clusters:
                clusters[cluster_label] = []
            clusters[cluster_label].append((conversations[i], label_bundles[i]))
        
        # Split clusters (not individual items)
        cluster_names = list(clusters.keys())
        random.shuffle(cluster_names)
        
        total_clusters = len(cluster_names)
        train_end = int(total_clusters * self.config.train_ratio)
        val_end = train_end + int(total_clusters * self.config.validation_ratio)
        
        train_clusters = cluster_names[:train_end]
        validation_clusters = cluster_names[train_end:val_end]
        test_clusters = cluster_names[val_end:]
        
        # Build the splits
        train = []
        validation = []
        test = []
        
        for cluster, items in clusters.items():
            if cluster in train_clusters:
                train.extend(items)
            elif cluster in validation_clusters:
                validation.extend(items)
            else:
                test.extend(items)
        
        # Calculate statistics
        paired_data = list(zip(conversations, label_bundles))
        split_stats = self._calculate_split_statistics(paired_data, (train, validation, test))
        
        return DatasetSplit(
            train=train,
            validation=validation,
            test=test,
            split_stats=split_stats
        )

    def _calculate_split_statistics(self, all_data: List[Tuple[Conversation, LabelBundle]], 
                                  splits: Tuple[List, List, List]) -> Dict[str, Any]:
        """Calculate statistics about the dataset splits - copy from Sampler class"""
        train, validation, test = splits
        
        stats = {
            "total_samples": len(all_data),
            "split_ratios": {
                "train": len(train) / len(all_data) if all_data else 0,
                "validation": len(validation) / len(all_data) if all_data else 0,
                "test": len(test) / len(all_data) if all_data else 0
            },
            "split_counts": {
                "train": len(train),
                "validation": len(validation),
                "test": len(test)
            },
            "label_distributions": {
                "train": self._calculate_label_distribution(train),
                "validation": self._calculate_label_distribution(validation),
                "test": self._calculate_label_distribution(test)
            },
            "imbalance_metrics": self._calculate_imbalance_metrics(splits)
        }
        
        return stats

    def _calculate_label_distribution(self, split_data: List[Tuple[Conversation, LabelBundle]]) -> Dict[str, Any]:
        """Calculate label distribution within a split - copy from Sampler class"""
        crisis_counts = Counter()
        response_counts = Counter()
        total_conversations = len(split_data)
        
        for conv, bundle in split_data:
            # Count crisis levels
            if bundle.crisis_label:
                crisis_counts[bundle.crisis_label.crisis_level.value] += 1
            else:
                crisis_counts["no_crisis"] += 1
            
            # Count therapeutic response types
            for label in bundle.therapeutic_response_labels:
                response_counts[label.response_type.value] += 1
        
        return {
            "crisis_distribution": {k: v/total_conversations if total_conversations > 0 else 0 
                                   for k, v in crisis_counts.items()},
            "response_distribution": {k: v/total_conversations if total_conversations > 0 else 0 
                                     for k, v in response_counts.items()},
            "total_conversations": total_conversations
        }

    def _calculate_imbalance_metrics(self, splits: Tuple[List, List, List]) -> Dict[str, float]:
        """Calculate metrics related to class imbalance across splits - copy from Sampler class"""
        train, validation, test = splits
        
        # Calculate imbalance for crisis levels
        train_dist = self._calculate_label_distribution(train)["crisis_distribution"]
        val_dist = self._calculate_label_distribution(validation)["crisis_distribution"]
        test_dist = self._calculate_label_distribution(test)["crisis_distribution"]
        
        # Calculate differences in distribution between splits
        max_crisis_diff = 0.0
        all_crisis_levels = set(train_dist.keys()).union(val_dist.keys()).union(test_dist.keys())
        
        for level in all_crisis_levels:
            train_ratio = train_dist.get(level, 0)
            val_ratio = val_dist.get(level, 0)
            test_ratio = test_dist.get(level, 0)
            
            max_diff = max(
                abs(train_ratio - val_ratio),
                abs(train_ratio - test_ratio),
                abs(val_ratio - test_ratio)
            )
            max_crisis_diff = max(max_crisis_diff, max_diff)
        
        return {
            "max_crisis_level_difference": max_crisis_diff
        }


def create_sampler(config: Optional[SplitConfig] = None) -> Sampler:
    """Create a default sampler with standard configuration"""
    return Sampler(config)


# Example usage
def test_sampling_utilities():
    """Test the sampling utilities"""
    from .conversation_schema import Conversation, Message
    from .label_taxonomy import (
        TherapeuticResponseLabel, CrisisLabel, LabelMetadata, LabelProvenanceType,
        TherapeuticResponseType, CrisisLevelType
    )
    
    # Create some test conversations and labels
    conversations = []
    label_bundles = []
    
    for i in range(50):
        # Create conversation
        conv = Conversation()
        conv.add_message("therapist", f"This is a sample therapeutic conversation {i}")
        conv.add_message("client", f"This is a client response {i}")
        conversations.append(conv)
        
        # Create corresponding label bundle
        bundle = LabelBundle(conversation_id=conv.conversation_id)
        
        # Add various therapeutic response labels
        response_type = random.choice(list(TherapeuticResponseType))
        bundle.therapeutic_response_labels.append(
            TherapeuticResponseLabel(
                response_type=response_type,
                metadata=LabelMetadata(confidence=random.uniform(0.7, 1.0))
            )
        )
        
        # Add crisis label
        crisis_level = random.choice(list(CrisisLevelType))
        bundle.crisis_label = CrisisLabel(
            crisis_level=crisis_level,
            metadata=LabelMetadata(confidence=random.uniform(0.6, 1.0))
        )
        
        label_bundles.append(bundle)
    
    print(f"Created {len(conversations)} conversations with corresponding labels")
    
    # Test basic split
    sampler = create_sampler()
    split_result = sampler.split_dataset(conversations, label_bundles)
    
    print(f"Split results:")
    print(f"  Train: {len(split_result.train)} samples")
    print(f"  Validation: {len(split_result.validation)} samples") 
    print(f"  Test: {len(split_result.test)} samples")
    print(f"  Stats: {split_result.split_stats}")
    
    # Test stratified split
    stratified_config = SplitConfig(
        split_type=SplitType.STRATIFIED,
        stratify_by="crisis_level",
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2
    )
    
    stratified_sampler = Sampler(stratified_config)
    stratified_split = stratified_sampler.split_dataset(conversations, label_bundles)
    
    print(f"\nStratified split results:")
    print(f"  Train: {len(stratified_split.train)} samples")
    print(f"  Validation: {len(stratified_split.validation)} samples")
    print(f"  Test: {len(stratified_split.test)} samples")
    print(f"  Crisis distribution in train: {stratified_split.split_stats['label_distributions']['train']['crisis_distribution']}")
    print(f"  Crisis distribution in validation: {stratified_split.split_stats['label_distributions']['validation']['crisis_distribution']}")
    print(f"  Crisis distribution in test: {stratified_split.split_stats['label_distributions']['test']['crisis_distribution']}")
    
    # Test balancing
    print(f"\nTesting dataset balancing...")
    balanced_convs, balanced_labels = sampler.balance_dataset(conversations, label_bundles, "oversample")
    print(f"  Original size: {len(conversations)}, Balanced size: {len(balanced_convs)}")
    
    # Test cluster-aware split
    print(f"\nTesting cluster-aware split...")
    cluster_labels = [f"cluster_{i % 5}" for i in range(len(conversations))]  # 5 clusters
    advanced_sampler = AdvancedSampler()
    
    try:
        cluster_split = advanced_sampler.cluster_aware_split(conversations, label_bundles, cluster_labels)
        print(f"  Cluster-aware split results:")
        print(f"    Train: {len(cluster_split.train)} samples")
        print(f"    Validation: {len(cluster_split.validation)} samples")
        print(f"    Test: {len(cluster_split.test)} samples")
    except Exception as e:
        print(f"  Cluster-aware split not fully implemented: {e}")


if __name__ == "__main__":
    test_sampling_utilities()