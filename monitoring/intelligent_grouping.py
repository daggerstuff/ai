#!/usr/bin/env python3
"""
Intelligent Alert Grouping Algorithms for Pixelated Empathy AI
Advanced pattern matching and machine learning-based alert grouping
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AlertFeatures:
    """Extracted features from an alert for grouping"""
    text_features: np.ndarray
    categorical_features: Dict[str, str]
    numerical_features: Dict[str, float]
    temporal_features: Dict[str, float]
    pattern_features: Dict[str, Any]

class TextPatternExtractor:
    """Extract patterns from alert text for intelligent grouping"""
    
    def __init__(self):
        self.common_patterns = [
            # Error patterns
            (r'error\s+code\s*:?\s*(\d+)', 'error_code'),
            (r'exception\s*:?\s*([A-Za-z][A-Za-z0-9_]*Exception)', 'exception_type'),
            (r'failed\s+to\s+([a-z_]+)', 'failure_action'),
            (r'timeout\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(ms|s|seconds?|minutes?)', 'timeout_duration'),
            
            # Resource patterns
            (r'cpu\s+usage\s*:?\s*(\d+(?:\.\d+)?)%?', 'cpu_usage'),
            (r'memory\s+usage\s*:?\s*(\d+(?:\.\d+)?)%?', 'memory_usage'),
            (r'disk\s+usage\s*:?\s*(\d+(?:\.\d+)?)%?', 'disk_usage'),
            (r'(\d+(?:\.\d+)?)\s*%\s+(?:cpu|memory|disk)', 'resource_percentage'),
            
            # Network patterns
            (r'connection\s+(?:to\s+)?([a-zA-Z0-9.-]+)(?:\s+port\s+(\d+))?\s+(?:failed|refused|timeout)', 'connection_target'),
            (r'(\d+\.\d+\.\d+\.\d+)(?::(\d+))?', 'ip_address'),
            (r'status\s+code\s*:?\s*(\d{3})', 'http_status'),
            
            # Service patterns
            (r'service\s+([a-zA-Z0-9_-]+)\s+(?:is\s+)?(?:down|unavailable|failed)', 'failed_service'),
            (r'([a-zA-Z0-9_-]+)\s+service\s+(?:is\s+)?(?:down|unavailable|failed)', 'failed_service'),
            
            # Database patterns
            (r'database\s+connection\s+(?:to\s+)?([a-zA-Z0-9_-]+)\s+failed', 'db_connection_target'),
            (r'query\s+timeout\s+(?:after\s+)?(\d+(?:\.\d+)?)\s*(ms|s|seconds?)', 'query_timeout'),
            
            # Queue patterns
            (r'queue\s+([a-zA-Z0-9_-]+)\s+(?:has\s+)?(\d+)\s+(?:pending\s+)?(?:items?|messages?)', 'queue_info'),
            (r'(\d+)\s+(?:items?|messages?)\s+(?:in\s+)?queue', 'queue_size'),
        ]
        
        self.severity_keywords = {
            'critical': ['critical', 'fatal', 'emergency', 'disaster'],
            'high': ['error', 'failed', 'failure', 'exception', 'crash'],
            'medium': ['warning', 'warn', 'degraded', 'slow', 'timeout'],
            'low': ['info', 'notice', 'debug', 'trace']
        }
    
    def extract_patterns(self, text: str) -> Dict[str, Any]:
        """Extract patterns from alert text"""
        if not text:
            return {}
        
        text_lower = text.lower()
        patterns = {}
        
        # Extract common patterns
        for pattern, pattern_name in self.common_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                if isinstance(matches[0], tuple):
                    patterns[pattern_name] = matches[0]
                else:
                    patterns[pattern_name] = matches[0]
        
        # Extract severity indicators
        for severity, keywords in self.severity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                patterns['inferred_severity'] = severity
                break
        
        # Extract numeric values
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            patterns['numeric_values'] = [float(n) for n in numbers[:5]]  # Limit to first 5
        
        # Extract identifiers (UUIDs, hashes, etc.)
        uuids = re.findall(r'\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b', text_lower)
        if uuids:
            patterns['has_uuid'] = True
        
        # Extract timestamps
        timestamps = re.findall(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}', text)
        if timestamps:
            patterns['has_timestamp'] = True
        
        return patterns

class AlertFeatureExtractor:
    """Extract comprehensive features from alerts for ML-based grouping"""
    
    def __init__(self):
        self.text_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.pattern_extractor = TextPatternExtractor()
        self.fitted = False
    
    def fit(self, alerts: List[Dict[str, Any]]):
        """Fit the feature extractor on a collection of alerts"""
        # Extract text content for TF-IDF fitting
        texts = []
        for alert in alerts:
            text_content = self._extract_text_content(alert)
            texts.append(text_content)
        
        if texts:
            self.text_vectorizer.fit(texts)
            self.fitted = True
            logger.info(f"Fitted feature extractor on {len(alerts)} alerts")
    
    def extract_features(self, alert: Dict[str, Any]) -> AlertFeatures:
        """Extract comprehensive features from an alert"""
        
        # Text features
        text_content = self._extract_text_content(alert)
        if self.fitted and text_content:
            text_features = self.text_vectorizer.transform([text_content]).toarray()[0]
        else:
            text_features = np.zeros(1000)  # Default size
        
        # Categorical features
        categorical_features = {
            'source': alert.get('source', 'unknown'),
            'alert_type': alert.get('alert_type', alert.get('title', 'unknown')),
            'priority': alert.get('priority', alert.get('severity', 'medium')),
            'service': alert.get('metadata', {}).get('service_name', 
                      alert.get('metadata', {}).get('service', 'unknown'))
        }
        
        # Numerical features
        numerical_features = {}
        metadata = alert.get('metadata', {})
        
        # Extract numerical values from metadata
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                numerical_features[key] = float(value)
            elif isinstance(value, str) and value.replace('.', '').isdigit():
                numerical_features[key] = float(value)
        
        # Temporal features
        timestamp = alert.get('timestamp', datetime.utcnow())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        temporal_features = {
            'hour_of_day': timestamp.hour,
            'day_of_week': timestamp.weekday(),
            'is_weekend': timestamp.weekday() >= 5,
            'is_business_hours': 9 <= timestamp.hour <= 17
        }
        
        # Pattern features
        message = alert.get('message', '')
        title = alert.get('title', '')
        full_text = f"{title} {message}"
        
        pattern_features = self.pattern_extractor.extract_patterns(full_text)
        
        return AlertFeatures(
            text_features=text_features,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            temporal_features=temporal_features,
            pattern_features=pattern_features
        )
    
    def _extract_text_content(self, alert: Dict[str, Any]) -> str:
        """Extract text content from alert for TF-IDF processing"""
        parts = []
        
        # Add title/alert_type
        if 'title' in alert:
            parts.append(alert['title'])
        elif 'alert_type' in alert:
            parts.append(alert['alert_type'])
        
        # Add message
        if 'message' in alert:
            parts.append(alert['message'])
        
        # Add relevant metadata
        metadata = alert.get('metadata', {})
        for key in ['description', 'details', 'error_message']:
            if key in metadata:
                parts.append(str(metadata[key]))
        
        return ' '.join(parts)

class IntelligentGroupingEngine:
    """Main engine for intelligent alert grouping using multiple algorithms"""
    
    def __init__(self):
        self.feature_extractor = AlertFeatureExtractor()
        self.grouping_algorithms = {
            'similarity_clustering': self._similarity_clustering,
            'pattern_matching': self._pattern_matching,
            'temporal_clustering': self._temporal_clustering,
            'hybrid_approach': self._hybrid_approach
        }
        self.alert_history = []
        self.group_cache = {}
    
    async def initialize(self, historical_alerts: List[Dict[str, Any]]):
        """Initialize the grouping engine with historical data"""
        if historical_alerts:
            self.alert_history = historical_alerts
            self.feature_extractor.fit(historical_alerts)
            logger.info(f"Initialized grouping engine with {len(historical_alerts)} historical alerts")
    
    async def suggest_groups(self, alerts: List[Dict[str, Any]], 
                           algorithm: str = 'hybrid_approach') -> List[List[int]]:
        """Suggest groupings for a list of alerts"""
        
        if not alerts:
            return []
        
        if algorithm not in self.grouping_algorithms:
            logger.warning(f"Unknown algorithm '{algorithm}', using hybrid_approach")
            algorithm = 'hybrid_approach'
        
        # Extract features for all alerts
        features = []
        for alert in alerts:
            alert_features = self.feature_extractor.extract_features(alert)
            features.append(alert_features)
        
        # Apply selected grouping algorithm
        grouping_func = self.grouping_algorithms[algorithm]
        groups = await grouping_func(alerts, features)
        
        logger.info(f"Grouped {len(alerts)} alerts into {len(groups)} groups using {algorithm}")
        return groups
    
    async def _similarity_clustering(self, alerts: List[Dict[str, Any]], 
                                   features: List[AlertFeatures]) -> List[List[int]]:
        """Group alerts using similarity-based clustering"""
        
        if len(alerts) < 2:
            return [[i] for i in range(len(alerts))]
        
        # Create similarity matrix using text features
        text_features = np.array([f.text_features for f in features])
        
        # Add categorical feature similarity
        categorical_similarity = self._calculate_categorical_similarity(features)
        
        # Combine similarities
        if text_features.shape[1] > 0:
            text_similarity = cosine_similarity(text_features)
            combined_similarity = 0.7 * text_similarity + 0.3 * categorical_similarity
        else:
            combined_similarity = categorical_similarity
        
        # Convert similarity to distance for DBSCAN
        distance_matrix = 1 - combined_similarity
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=0.3,  # Maximum distance between samples in the same cluster
            min_samples=2,  # Minimum samples in a cluster
            metric='precomputed'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Convert cluster labels to groups
        groups = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise points get their own groups
                groups[f"noise_{i}"] = [i]
            else:
                groups[label].append(i)
        
        return list(groups.values())
    
    def _calculate_categorical_similarity(self, features: List[AlertFeatures]) -> np.ndarray:
        """Calculate similarity matrix based on categorical features"""
        
        n = len(features)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Calculate categorical similarity
                    cat_i = features[i].categorical_features
                    cat_j = features[j].categorical_features
                    
                    matches = 0
                    total = 0
                    
                    for key in set(cat_i.keys()) | set(cat_j.keys()):
                        total += 1
                        if key in cat_i and key in cat_j and cat_i[key] == cat_j[key]:
                            matches += 1
                    
                    similarity_matrix[i][j] = matches / total if total > 0 else 0.0
        
        return similarity_matrix
    
    async def _pattern_matching(self, alerts: List[Dict[str, Any]], 
                              features: List[AlertFeatures]) -> List[List[int]]:
        """Group alerts using pattern matching"""
        
        groups = defaultdict(list)
        
        for i, feature in enumerate(features):
            # Create a pattern signature
            pattern_sig = self._create_pattern_signature(feature)
            groups[pattern_sig].append(i)
        
        # Filter out single-item groups and merge similar patterns
        filtered_groups = []
        pattern_groups = list(groups.values())
        
        for group in pattern_groups:
            if len(group) >= 2:  # Only keep groups with multiple alerts
                filtered_groups.append(group)
            else:
                # Try to merge with existing groups
                merged = False
                for existing_group in filtered_groups:
                    if self._should_merge_groups(
                        [features[group[0]]], 
                        [features[existing_group[0]]]
                    ):
                        existing_group.extend(group)
                        merged = True
                        break
                
                if not merged:
                    filtered_groups.append(group)
        
        return filtered_groups
    
    def _create_pattern_signature(self, feature: AlertFeatures) -> str:
        """Create a signature for pattern matching"""
        
        sig_parts = []
        
        # Add categorical features
        cat_features = feature.categorical_features
        sig_parts.append(f"src:{cat_features.get('source', 'unknown')}")
        sig_parts.append(f"type:{cat_features.get('alert_type', 'unknown')}")
        sig_parts.append(f"svc:{cat_features.get('service', 'unknown')}")
        
        # Add pattern features
        pattern_features = feature.pattern_features
        if 'error_code' in pattern_features:
            sig_parts.append(f"err:{pattern_features['error_code']}")
        if 'exception_type' in pattern_features:
            sig_parts.append(f"exc:{pattern_features['exception_type']}")
        if 'inferred_severity' in pattern_features:
            sig_parts.append(f"sev:{pattern_features['inferred_severity']}")
        
        # Add temporal context
        temporal = feature.temporal_features
        if temporal.get('is_business_hours'):
            sig_parts.append("time:business")
        else:
            sig_parts.append("time:off")
        
        return "|".join(sorted(sig_parts))
    
    def _should_merge_groups(self, group1_features: List[AlertFeatures], 
                           group2_features: List[AlertFeatures]) -> bool:
        """Determine if two groups should be merged"""
        
        # Sample features from each group
        f1 = group1_features[0]
        f2 = group2_features[0]
        
        # Check categorical similarity
        cat_similarity = 0
        cat_total = 0
        
        for key in set(f1.categorical_features.keys()) | set(f2.categorical_features.keys()):
            cat_total += 1
            if (key in f1.categorical_features and key in f2.categorical_features and
                f1.categorical_features[key] == f2.categorical_features[key]):
                cat_similarity += 1
        
        categorical_score = cat_similarity / cat_total if cat_total > 0 else 0
        
        # Check pattern similarity
        pattern_similarity = len(
            set(f1.pattern_features.keys()) & set(f2.pattern_features.keys())
        ) / max(len(f1.pattern_features), len(f2.pattern_features), 1)
        
        # Merge if similarity is high enough
        return categorical_score >= 0.7 and pattern_similarity >= 0.5
    
    async def _temporal_clustering(self, alerts: List[Dict[str, Any]], 
                                 features: List[AlertFeatures]) -> List[List[int]]:
        """Group alerts using temporal clustering"""
        
        # Sort alerts by timestamp
        alert_times = []
        for i, alert in enumerate(alerts):
            timestamp = alert.get('timestamp', datetime.utcnow())
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            alert_times.append((i, timestamp))
        
        alert_times.sort(key=lambda x: x[1])
        
        # Group alerts within time windows
        groups = []
        current_group = []
        time_window = timedelta(minutes=5)  # 5-minute window
        
        for i, (alert_idx, timestamp) in enumerate(alert_times):
            if not current_group:
                current_group = [alert_idx]
            else:
                # Check if within time window of the first alert in current group
                first_timestamp = alert_times[alert_times.index((current_group[0], 
                    next(t for idx, t in alert_times if idx == current_group[0])))][1]
                
                if timestamp - first_timestamp <= time_window:
                    current_group.append(alert_idx)
                else:
                    # Start new group
                    if len(current_group) >= 2:
                        groups.append(current_group)
                    current_group = [alert_idx]
        
        # Add the last group
        if len(current_group) >= 2:
            groups.append(current_group)
        
        # Add single alerts as individual groups
        grouped_indices = set()
        for group in groups:
            grouped_indices.update(group)
        
        for i in range(len(alerts)):
            if i not in grouped_indices:
                groups.append([i])
        
        return groups
    
    async def _hybrid_approach(self, alerts: List[Dict[str, Any]], 
                             features: List[AlertFeatures]) -> List[List[int]]:
        """Hybrid approach combining multiple grouping strategies"""
        
        # Start with similarity clustering
        similarity_groups = await self._similarity_clustering(alerts, features)
        
        # Refine with pattern matching
        refined_groups = []
        for group in similarity_groups:
            if len(group) <= 2:
                refined_groups.append(group)
                continue
            
            # Apply pattern matching within the similarity group
            group_alerts = [alerts[i] for i in group]
            group_features = [features[i] for i in group]
            
            pattern_subgroups = await self._pattern_matching(group_alerts, group_features)
            
            # Map back to original indices
            for subgroup in pattern_subgroups:
                original_indices = [group[i] for i in subgroup]
                refined_groups.append(original_indices)
        
        # Apply temporal refinement
        final_groups = []
        for group in refined_groups:
            if len(group) <= 3:
                final_groups.append(group)
                continue
            
            # Check temporal distribution within group
            group_alerts = [alerts[i] for i in group]
            group_features = [features[i] for i in group]
            
            temporal_subgroups = await self._temporal_clustering(group_alerts, group_features)
            
            # Map back to original indices
            for subgroup in temporal_subgroups:
                original_indices = [group[i] for i in subgroup]
                final_groups.append(original_indices)
        
        return final_groups
    
    async def evaluate_grouping_quality(self, alerts: List[Dict[str, Any]], 
                                      groups: List[List[int]]) -> Dict[str, float]:
        """Evaluate the quality of alert grouping"""
        
        if not groups or not alerts:
            return {"silhouette_score": 0.0, "cohesion": 0.0, "separation": 0.0}
        
        # Extract features
        features = [self.feature_extractor.extract_features(alert) for alert in alerts]
        
        # Calculate intra-group cohesion
        cohesion_scores = []
        for group in groups:
            if len(group) < 2:
                continue
            
            group_features = [features[i].text_features for i in group]
            if len(group_features) > 1:
                similarity_matrix = cosine_similarity(group_features)
                # Average similarity within group (excluding diagonal)
                mask = np.ones_like(similarity_matrix, dtype=bool)
                np.fill_diagonal(mask, False)
                cohesion = similarity_matrix[mask].mean()
                cohesion_scores.append(cohesion)
        
        avg_cohesion = np.mean(cohesion_scores) if cohesion_scores else 0.0
        
        # Calculate inter-group separation
        separation_scores = []
        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups[i+1:], i+1):
                if len(group1) == 0 or len(group2) == 0:
                    continue
                
                # Calculate average similarity between groups
                group1_features = [features[idx].text_features for idx in group1]
                group2_features = [features[idx].text_features for idx in group2]
                
                if group1_features and group2_features:
                    cross_similarity = cosine_similarity(group1_features, group2_features)
                    separation_scores.append(1 - cross_similarity.mean())  # 1 - similarity = dissimilarity
        
        avg_separation = np.mean(separation_scores) if separation_scores else 1.0
        
        # Simple silhouette-like score
        silhouette_score = (avg_cohesion + avg_separation) / 2
        
        return {
            "silhouette_score": float(silhouette_score),
            "cohesion": float(avg_cohesion),
            "separation": float(avg_separation),
            "num_groups": len(groups),
            "avg_group_size": np.mean([len(g) for g in groups]) if groups else 0.0
        }

# Example usage and testing
async def example_usage():
    """Example of how to use the intelligent grouping system"""
    
    # Initialize grouping engine
    engine = IntelligentGroupingEngine()
    
    # Example alerts
    test_alerts = [
        {
            "title": "High CPU Usage",
            "message": "CPU usage is 87% on server-01",
            "priority": "medium",
            "source": "monitoring",
            "timestamp": "2025-08-27T00:30:00Z",
            "metadata": {"server": "server-01", "cpu_usage": 87}
        },
        {
            "title": "High CPU Usage",
            "message": "CPU usage is 89% on server-01", 
            "priority": "medium",
            "source": "monitoring",
            "timestamp": "2025-08-27T00:32:00Z",
            "metadata": {"server": "server-01", "cpu_usage": 89}
        },
        {
            "title": "Database Connection Failed",
            "message": "Connection to database db-prod failed with timeout",
            "priority": "high",
            "source": "application",
            "timestamp": "2025-08-27T00:31:00Z",
            "metadata": {"database": "db-prod", "error": "timeout"}
        },
        {
            "title": "Database Connection Failed", 
            "message": "Connection to database db-prod failed with timeout",
            "priority": "high",
            "source": "application",
            "timestamp": "2025-08-27T00:33:00Z",
            "metadata": {"database": "db-prod", "error": "timeout"}
        },
        {
            "title": "Memory Usage Warning",
            "message": "Memory usage is 82% on server-02",
            "priority": "medium", 
            "source": "monitoring",
            "timestamp": "2025-08-27T00:35:00Z",
            "metadata": {"server": "server-02", "memory_usage": 82}
        }
    ]
    
    # Initialize with historical data (in practice, this would be real historical alerts)
    await engine.initialize(test_alerts[:3])
    
    # Test different grouping algorithms
    algorithms = ['similarity_clustering', 'pattern_matching', 'temporal_clustering', 'hybrid_approach']
    
    for algorithm in algorithms:
        print(f"\n=== {algorithm.upper()} ===")
        groups = await engine.suggest_groups(test_alerts, algorithm)
        
        print(f"Found {len(groups)} groups:")
        for i, group in enumerate(groups):
            print(f"  Group {i+1}: {group}")
            for idx in group:
                alert = test_alerts[idx]
                print(f"    - {alert['title']} ({alert['source']})")
        
        # Evaluate grouping quality
        quality = await engine.evaluate_grouping_quality(test_alerts, groups)
        print(f"Quality metrics: {quality}")

if __name__ == "__main__":
    asyncio.run(example_usage())
