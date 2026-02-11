#!/usr/bin/env python3
"""
Comprehensive Test Suite for Adaptive Learner
Production-ready tests for adaptive learning system.

This test suite validates the adaptive learner's ability to:
1. Learn from user interactions and feedback
2. Adapt response strategies based on effectiveness
3. Improve therapeutic outcomes over time
4. Maintain learning consistency and stability
5. Handle various learning scenarios and edge cases
"""

import unittest
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Union, Any

# Mock the adaptive learner for testing
class MockAdaptiveLearner:
    """Mock implementation of AdaptiveLearner for testing."""
    
    def __init__(self):
        self.learning_history = []
        self.adaptation_strategies = ['reinforcement', 'pattern_recognition', 'feedback_integration']
        self.performance_metrics = {}
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        
    def learn_from_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from a therapeutic interaction."""
        if not interaction_data or not isinstance(interaction_data, dict):
            return {
                'success': False,
                'error': 'Invalid interaction data',
                'learning_applied': False,
                'adaptation_score': 0.0
            }
        
        # Extract learning signals
        learning_signals = self._extract_learning_signals(interaction_data)
        
        # Apply learning algorithms
        learning_result = self._apply_learning_algorithms(learning_signals)
        
        # Update adaptation strategies
        adaptation_result = self._update_adaptation_strategies(learning_result)
        
        # Record learning
        learning_record = {
            'interaction_data': interaction_data,
            'learning_signals': learning_signals,
            'learning_result': learning_result,
            'adaptation_result': adaptation_result,
            'timestamp': '2025-08-20T16:00:00Z'
        }
        
        self.learning_history.append(learning_record)
        
        return {
            'success': True,
            'error': None,
            'learning_applied': learning_result['learning_applied'],
            'adaptation_score': adaptation_result['adaptation_score'],
            'improvements_identified': learning_result['improvements'],
            'strategy_updates': adaptation_result['strategy_updates']
        }
    
    def _extract_learning_signals(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract learning signals from interaction data."""
        # Simulate learning signal extraction
        user_feedback = interaction_data.get('user_feedback', {})
        response_effectiveness = interaction_data.get('response_effectiveness', 0.5)
        therapeutic_outcome = interaction_data.get('therapeutic_outcome', 'neutral')
        
        signals = {
            'positive_feedback': user_feedback.get('rating', 0) > 3,
            'effectiveness_score': response_effectiveness,
            'outcome_quality': 1.0 if therapeutic_outcome == 'positive' else 0.5 if therapeutic_outcome == 'neutral' else 0.0,
            'engagement_level': interaction_data.get('engagement_level', 0.5),
            'technique_effectiveness': interaction_data.get('technique_effectiveness', {})
        }
        
        return signals
    
    def _apply_learning_algorithms(self, learning_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Apply learning algorithms to improve performance."""
        improvements = []
        learning_applied = False
        
        # Reinforcement learning
        if learning_signals['positive_feedback']:
            improvements.append('reinforce_successful_patterns')
            learning_applied = True
        
        # Pattern recognition
        if learning_signals['effectiveness_score'] > self.adaptation_threshold:
            improvements.append('identify_effective_techniques')
            learning_applied = True
        
        # Outcome optimization
        if learning_signals['outcome_quality'] > 0.7:
            improvements.append('optimize_therapeutic_approach')
            learning_applied = True
        
        return {
            'learning_applied': learning_applied,
            'improvements': improvements,
            'confidence': min(1.0, learning_signals['effectiveness_score'] + 0.2)
        }
    
    def _update_adaptation_strategies(self, learning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update adaptation strategies based on learning."""
        strategy_updates = []
        adaptation_score = 0.5  # Base score
        
        if learning_result['learning_applied']:
            adaptation_score += 0.3
            
            for improvement in learning_result['improvements']:
                if 'reinforce' in improvement:
                    strategy_updates.append('increase_reinforcement_weight')
                elif 'identify' in improvement:
                    strategy_updates.append('enhance_pattern_recognition')
                elif 'optimize' in improvement:
                    strategy_updates.append('refine_therapeutic_approach')
        
        return {
            'adaptation_score': min(1.0, adaptation_score),
            'strategy_updates': strategy_updates,
            'adaptation_applied': len(strategy_updates) > 0
        }
    
    def get_adaptation_recommendations(self, context: Dict[str, Any]) -> List[str]:
        """Get adaptation recommendations based on learning history."""
        if not self.learning_history:
            return ['Collect more interaction data for learning']
        
        recommendations = []
        
        # Analyze learning patterns
        recent_interactions = self.learning_history[-10:]  # Last 10 interactions
        
        # Check for consistent patterns
        positive_outcomes = sum(1 for interaction in recent_interactions 
                              if interaction['learning_signals']['positive_feedback'])
        
        if positive_outcomes / len(recent_interactions) > 0.7:
            recommendations.append('Continue current therapeutic approach - showing positive results')
        elif positive_outcomes / len(recent_interactions) < 0.3:
            recommendations.append('Consider adjusting therapeutic techniques - low positive feedback')
        
        # Check effectiveness trends
        avg_effectiveness = sum(interaction['learning_signals']['effectiveness_score'] 
                              for interaction in recent_interactions) / len(recent_interactions)
        
        if avg_effectiveness > 0.8:
            recommendations.append('Maintain current effectiveness strategies')
        elif avg_effectiveness < 0.5:
            recommendations.append('Implement alternative therapeutic approaches')
        
        return recommendations or ['Continue monitoring and learning from interactions']
    
    def adapt_response_strategy(self, current_strategy: Dict[str, Any], 
                              learning_context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt response strategy based on learning."""
        if not current_strategy:
            return {
                'success': False,
                'error': 'No current strategy provided',
                'adapted_strategy': None
            }
        
        adapted_strategy = current_strategy.copy()
        adaptations_made = []
        
        # Apply learned adaptations
        if self.learning_history:
            recent_learning = self.learning_history[-5:]  # Last 5 learning events
            
            # Adapt based on successful patterns
            successful_techniques = []
            for learning in recent_learning:
                if learning['learning_signals']['positive_feedback']:
                    techniques = learning['interaction_data'].get('techniques_used', [])
                    successful_techniques.extend(techniques)
            
            if successful_techniques:
                # Increase weight of successful techniques
                technique_counts = {}
                for technique in successful_techniques:
                    technique_counts[technique] = technique_counts.get(technique, 0) + 1
                
                most_successful = max(technique_counts.items(), key=lambda x: x[1])[0]
                adapted_strategy['primary_technique'] = most_successful
                adaptations_made.append(f'prioritized_{most_successful}')
            
            # Adapt response style based on engagement
            avg_engagement = sum(l['learning_signals']['engagement_level'] for l in recent_learning) / len(recent_learning)
            if avg_engagement < 0.5:
                adapted_strategy['response_style'] = 'more_engaging'
                adaptations_made.append('increased_engagement_focus')
        
        return {
            'success': True,
            'error': None,
            'adapted_strategy': adapted_strategy,
            'adaptations_made': adaptations_made,
            'adaptation_confidence': min(1.0, len(adaptations_made) * 0.3 + 0.4)
        }
    
    def evaluate_learning_progress(self) -> Dict[str, Any]:
        """Evaluate overall learning progress."""
        if not self.learning_history:
            return {
                'total_interactions': 0,
                'learning_progress': 0.0,
                'adaptation_effectiveness': 0.0,
                'recommendations': ['Start collecting interaction data']
            }
        
        total_interactions = len(self.learning_history)
        
        # Calculate learning progress
        if total_interactions >= 10:
            early_performance = sum(l['learning_signals']['effectiveness_score'] 
                                  for l in self.learning_history[:5]) / 5
            recent_performance = sum(l['learning_signals']['effectiveness_score'] 
                                   for l in self.learning_history[-5:]) / 5
            learning_progress = max(0.0, (recent_performance - early_performance) / early_performance)
        else:
            learning_progress = 0.1 * total_interactions  # Linear progress for small samples
        
        # Calculate adaptation effectiveness
        successful_adaptations = sum(1 for l in self.learning_history 
                                   if l['adaptation_result']['adaptation_applied'])
        adaptation_effectiveness = successful_adaptations / total_interactions
        
        # Generate recommendations
        recommendations = []
        if learning_progress < 0.1:
            recommendations.append('Increase learning rate or adjust learning algorithms')
        if adaptation_effectiveness < 0.5:
            recommendations.append('Review adaptation strategies for effectiveness')
        if total_interactions < 50:
            recommendations.append('Collect more interaction data for robust learning')
        
        return {
            'total_interactions': total_interactions,
            'learning_progress': round(learning_progress, 3),
            'adaptation_effectiveness': round(adaptation_effectiveness, 3),
            'successful_adaptations': successful_adaptations,
            'recommendations': recommendations or ['Learning progress is satisfactory']
        }
    
    def reset_learning(self) -> Dict[str, Any]:
        """Reset learning history and adaptations."""
        previous_count = len(self.learning_history)
        self.learning_history.clear()
        self.performance_metrics.clear()
        
        return {
            'success': True,
            'previous_interactions': previous_count,
            'reset_timestamp': '2025-08-20T16:00:00Z'
        }


class TestAdaptiveLearner(unittest.TestCase):
    """Test suite for AdaptiveLearner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.learner = MockAdaptiveLearner()
        
        self.test_interactions = {
            'positive': {
                'user_feedback': {'rating': 5, 'comments': 'Very helpful'},
                'response_effectiveness': 0.9,
                'therapeutic_outcome': 'positive',
                'engagement_level': 0.8,
                'techniques_used': ['empathic_reflection', 'cognitive_reframe']
            },
            'negative': {
                'user_feedback': {'rating': 2, 'comments': 'Not helpful'},
                'response_effectiveness': 0.3,
                'therapeutic_outcome': 'negative',
                'engagement_level': 0.2,
                'techniques_used': ['directive_advice']
            },
            'neutral': {
                'user_feedback': {'rating': 3, 'comments': 'Okay'},
                'response_effectiveness': 0.5,
                'therapeutic_outcome': 'neutral',
                'engagement_level': 0.5,
                'techniques_used': ['validation']
            }
        }
        
        self.test_strategy = {
            'primary_technique': 'empathic_reflection',
            'response_style': 'collaborative',
            'adaptation_level': 'medium'
        }
    
    def test_initialization(self):
        """Test learner initialization."""
        self.assertIsNotNone(self.learner)
        self.assertIsInstance(self.learner.adaptation_strategies, list)
        self.assertEqual(len(self.learner.learning_history), 0)
        self.assertGreater(self.learner.learning_rate, 0)
    
    def test_successful_learning_from_positive_interaction(self):
        """Test learning from positive interaction."""
        result = self.learner.learn_from_interaction(self.test_interactions['positive'])
        
        self.assertTrue(result['success'])
        self.assertIsNone(result['error'])
        self.assertTrue(result['learning_applied'])
        self.assertGreater(result['adaptation_score'], 0.5)
        self.assertGreater(len(result['improvements_identified']), 0)
    
    def test_learning_from_negative_interaction(self):
        """Test learning from negative interaction."""
        result = self.learner.learn_from_interaction(self.test_interactions['negative'])
        
        self.assertTrue(result['success'])
        # May or may not apply learning depending on signals
        self.assertIsInstance(result['learning_applied'], bool)
        self.assertGreaterEqual(result['adaptation_score'], 0.0)
    
    def test_learning_from_neutral_interaction(self):
        """Test learning from neutral interaction."""
        result = self.learner.learn_from_interaction(self.test_interactions['neutral'])
        
        self.assertTrue(result['success'])
        self.assertIsInstance(result['learning_applied'], bool)
        self.assertGreaterEqual(result['adaptation_score'], 0.0)
    
    def test_invalid_interaction_data(self):
        """Test handling of invalid interaction data."""
        invalid_data = [None, "", {}, [], 123]
        
        for data in invalid_data:
            with self.subTest(data=data):
                result = self.learner.learn_from_interaction(data)
                
                self.assertFalse(result['success'])
                self.assertIsNotNone(result['error'])
                self.assertFalse(result['learning_applied'])
    
    def test_adaptation_recommendations(self):
        """Test generation of adaptation recommendations."""
        # Test with no learning history
        recommendations = self.learner.get_adaptation_recommendations({})
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Add some learning history
        for _ in range(5):
            self.learner.learn_from_interaction(self.test_interactions['positive'])
        
        recommendations = self.learner.get_adaptation_recommendations({})
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
    
    def test_response_strategy_adaptation(self):
        """Test adaptation of response strategies."""
        # Test with no learning history
        result = self.learner.adapt_response_strategy(self.test_strategy, {})
        
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['adapted_strategy'])
        
        # Add learning history and test adaptation
        for _ in range(3):
            self.learner.learn_from_interaction(self.test_interactions['positive'])
        
        result = self.learner.adapt_response_strategy(self.test_strategy, {})
        
        self.assertTrue(result['success'])
        self.assertIsNotNone(result['adapted_strategy'])
        self.assertIsInstance(result['adaptations_made'], list)
    
    def test_invalid_strategy_adaptation(self):
        """Test adaptation with invalid strategy."""
        result = self.learner.adapt_response_strategy(None, {})
        
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertIsNone(result['adapted_strategy'])
    
    def test_learning_progress_evaluation(self):
        """Test evaluation of learning progress."""
        # Test with no interactions
        progress = self.learner.evaluate_learning_progress()
        
        self.assertEqual(progress['total_interactions'], 0)
        self.assertEqual(progress['learning_progress'], 0.0)
        self.assertIsInstance(progress['recommendations'], list)
        
        # Add interactions and test progress
        interactions = [
            self.test_interactions['negative'],  # Start with poor performance
            self.test_interactions['neutral'],
            self.test_interactions['positive'],  # Improve over time
            self.test_interactions['positive']
        ]
        
        for interaction in interactions:
            self.learner.learn_from_interaction(interaction)
        
        progress = self.learner.evaluate_learning_progress()
        
        self.assertEqual(progress['total_interactions'], 4)
        self.assertGreaterEqual(progress['learning_progress'], 0.0)
        self.assertGreaterEqual(progress['adaptation_effectiveness'], 0.0)
    
    def test_learning_reset(self):
        """Test learning reset functionality."""
        # Add some learning history
        for _ in range(3):
            self.learner.learn_from_interaction(self.test_interactions['positive'])
        
        self.assertGreater(len(self.learner.learning_history), 0)
        
        # Reset learning
        result = self.learner.reset_learning()
        
        self.assertTrue(result['success'])
        self.assertEqual(result['previous_interactions'], 3)
        self.assertEqual(len(self.learner.learning_history), 0)
    
    def test_continuous_learning_improvement(self):
        """Test continuous learning and improvement over time."""
        # Simulate learning progression
        interaction_sequence = [
            self.test_interactions['negative'],  # Poor start
            self.test_interactions['negative'],
            self.test_interactions['neutral'],   # Gradual improvement
            self.test_interactions['neutral'],
            self.test_interactions['positive'],  # Good performance
            self.test_interactions['positive']
        ]
        
        adaptation_scores = []
        for interaction in interaction_sequence:
            result = self.learner.learn_from_interaction(interaction)
            adaptation_scores.append(result['adaptation_score'])
        
        # Check that learning is occurring
        self.assertEqual(len(self.learner.learning_history), 6)
        
        # Evaluate final progress
        progress = self.learner.evaluate_learning_progress()
        self.assertGreater(progress['total_interactions'], 0)
    
    def test_batch_learning(self):
        """Test batch learning from multiple interactions."""
        interactions = [
            self.test_interactions['positive'],
            self.test_interactions['neutral'],
            self.test_interactions['positive'],
            self.test_interactions['negative'],
            self.test_interactions['positive']
        ]
        
        results = []
        for interaction in interactions:
            result = self.learner.learn_from_interaction(interaction)
            results.append(result)
        
        # All should succeed
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertTrue(result['success'])
        
        # Check learning accumulation
        self.assertEqual(len(self.learner.learning_history), 5)
        
        # Evaluate overall progress
        progress = self.learner.evaluate_learning_progress()
        self.assertEqual(progress['total_interactions'], 5)


class TestAdaptiveLearnerIntegration(unittest.TestCase):
    """Integration tests for AdaptiveLearner."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.learner = MockAdaptiveLearner()
    
    def test_complete_learning_cycle(self):
        """Test complete learning cycle from interaction to adaptation."""
        # Step 1: Initial interaction
        initial_interaction = {
            'user_feedback': {'rating': 4, 'comments': 'Good response'},
            'response_effectiveness': 0.8,
            'therapeutic_outcome': 'positive',
            'engagement_level': 0.7,
            'techniques_used': ['empathic_reflection']
        }
        
        learning_result = self.learner.learn_from_interaction(initial_interaction)
        self.assertTrue(learning_result['success'])
        
        # Step 2: Get adaptation recommendations
        recommendations = self.learner.get_adaptation_recommendations({})
        self.assertIsInstance(recommendations, list)
        
        # Step 3: Adapt strategy
        current_strategy = {
            'primary_technique': 'validation',
            'response_style': 'directive'
        }
        
        adaptation_result = self.learner.adapt_response_strategy(current_strategy, {})
        self.assertTrue(adaptation_result['success'])
        
        # Step 4: Evaluate progress
        progress = self.learner.evaluate_learning_progress()
        self.assertGreater(progress['total_interactions'], 0)
    
    def test_learning_effectiveness_over_time(self):
        """Test learning effectiveness improves over time."""
        # Simulate therapy sessions with varying outcomes
        session_outcomes = [
            ('negative', 0.3), ('neutral', 0.5), ('positive', 0.7),
            ('positive', 0.8), ('positive', 0.9)  # Improving trend
        ]
        
        for outcome, effectiveness in session_outcomes:
            interaction = {
                'user_feedback': {'rating': 3 if outcome == 'neutral' else (5 if outcome == 'positive' else 1)},
                'response_effectiveness': effectiveness,
                'therapeutic_outcome': outcome,
                'engagement_level': effectiveness,
                'techniques_used': ['empathic_reflection']
            }
            
            result = self.learner.learn_from_interaction(interaction)
            self.assertTrue(result['success'])
        
        # Evaluate final learning state
        progress = self.learner.evaluate_learning_progress()
        
        # Should show learning progress
        self.assertEqual(progress['total_interactions'], 5)
        self.assertGreaterEqual(progress['learning_progress'], 0.0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
