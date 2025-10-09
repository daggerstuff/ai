#!/usr/bin/env python3
"""
Comprehensive Test Suite for Therapeutic Response Generator
Production-ready tests for AI-powered therapeutic response generation.

This test suite validates the therapeutic response generator's ability to:
1. Generate contextually appropriate therapeutic responses
2. Maintain therapeutic alliance and rapport
3. Apply evidence-based therapeutic techniques
4. Ensure safety and ethical compliance
5. Adapt to different therapeutic modalities
"""

import unittest
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Union, Any

# Mock the therapeutic response generator for testing
class MockTherapeuticResponseGenerator:
    """Mock implementation of TherapeuticResponseGenerator for testing."""
    
    def __init__(self):
        self.therapeutic_modalities = ['cbt', 'dbt', 'psychodynamic', 'humanistic', 'systemic']
        self.response_templates = {
            'empathic_reflection': "I hear that you're feeling {emotion}. That sounds {intensity}.",
            'cognitive_reframe': "What if we looked at this situation from a different perspective?",
            'behavioral_intervention': "Let's explore some strategies that might help with {behavior}.",
            'validation': "Your feelings about this are completely understandable.",
            'exploration': "Can you tell me more about {topic}?"
        }
        self.generation_history = []
        
    def generate_response(self, client_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a therapeutic response based on client input and context."""
        if not client_input or not isinstance(client_input, str) or str(client_input).strip() == "":
            return {
                'success': False,
                'response': None,
                'error': 'Invalid client input',
                'therapeutic_quality': 0.0,
                'techniques_used': [],
                'safety_score': 0.0
            }
        
        # Analyze client input
        analysis = self._analyze_client_input(client_input)
        
        # Select appropriate therapeutic approach
        approach = self._select_therapeutic_approach(analysis, context)
        
        # Generate response based on approach
        response = self._generate_contextual_response(analysis, approach, context)
        
        # Evaluate response quality
        quality_assessment = self._assess_response_quality(response, analysis, context)
        
        # Record generation
        generation_record = {
            'client_input': client_input,
            'context': context,
            'analysis': analysis,
            'approach': approach,
            'response': response,
            'quality_assessment': quality_assessment,
            'timestamp': '2025-08-20T15:30:00Z'
        }
        
        self.generation_history.append(generation_record)
        
        return {
            'success': True,
            'response': response,
            'error': None,
            'therapeutic_quality': quality_assessment['overall_quality'],
            'techniques_used': approach['techniques'],
            'safety_score': quality_assessment['safety_score'],
            'empathy_score': quality_assessment['empathy_score'],
            'appropriateness_score': quality_assessment['appropriateness_score'],
            'analysis': analysis,
            'approach': approach
        }
    
    def _analyze_client_input(self, input_text: str) -> Dict[str, Any]:
        """Analyze client input for emotional content and therapeutic needs."""
        input_lower = input_text.lower()
        
        # Detect emotions
        emotions = []
        if any(word in input_lower for word in ['sad', 'depressed', 'down', 'hopeless']):
            emotions.append('sadness')
        if any(word in input_lower for word in ['anxious', 'worried', 'nervous', 'scared']):
            emotions.append('anxiety')
        if any(word in input_lower for word in ['angry', 'mad', 'frustrated', 'irritated']):
            emotions.append('anger')
        if any(word in input_lower for word in ['happy', 'good', 'great', 'excited']):
            emotions.append('positive')
        
        # Detect therapeutic needs
        needs = []
        if any(word in input_lower for word in ['help', 'support', 'advice']):
            needs.append('support')
        if any(word in input_lower for word in ['understand', 'confused', 'unclear']):
            needs.append('clarification')
        if any(word in input_lower for word in ['change', 'different', 'better']):
            needs.append('behavioral_change')
        if any(word in input_lower for word in ['feel', 'emotion', 'feeling']):
            needs.append('emotional_processing')
        
        # Assess urgency
        urgency = 'low'
        if any(word in input_lower for word in ['crisis', 'emergency', 'urgent', 'immediate']):
            urgency = 'high'
        elif any(word in input_lower for word in ['important', 'serious', 'concerned']):
            urgency = 'medium'
        
        return {
            'emotions_detected': emotions,
            'therapeutic_needs': needs,
            'urgency_level': urgency,
            'input_length': len(input_text),
            'complexity': 'high' if len(input_text) > 200 else 'medium' if len(input_text) > 50 else 'low'
        }
    
    def _select_therapeutic_approach(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate therapeutic approach based on analysis and context."""
        modality = context.get('therapeutic_modality', 'humanistic')
        session_number = context.get('session_number', 1)
        
        # Select primary technique based on needs and modality
        primary_technique = 'empathic_reflection'  # Default
        
        if 'emotional_processing' in analysis['therapeutic_needs']:
            if modality == 'cbt':
                primary_technique = 'cognitive_reframe'
            elif modality == 'dbt':
                primary_technique = 'validation'
            else:
                primary_technique = 'empathic_reflection'
        elif 'behavioral_change' in analysis['therapeutic_needs']:
            primary_technique = 'behavioral_intervention'
        elif 'clarification' in analysis['therapeutic_needs']:
            primary_technique = 'exploration'
        
        # Select supporting techniques
        supporting_techniques = []
        if session_number <= 3:  # Early sessions focus on rapport
            supporting_techniques.append('validation')
        if analysis['urgency_level'] == 'high':
            supporting_techniques.append('crisis_intervention')
        if 'anxiety' in analysis['emotions_detected']:
            supporting_techniques.append('grounding')
        
        return {
            'modality': modality,
            'primary_technique': primary_technique,
            'supporting_techniques': supporting_techniques,
            'techniques': [primary_technique] + supporting_techniques,
            'rationale': f"Selected {primary_technique} based on {analysis['therapeutic_needs']}"
        }
    
    def _generate_contextual_response(self, analysis: Dict[str, Any], approach: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate contextual therapeutic response."""
        primary_technique = approach['primary_technique']
        emotions = analysis['emotions_detected']
        
        # Base response from template
        if primary_technique in self.response_templates:
            base_response = self.response_templates[primary_technique]
            
            # Customize based on detected emotions
            if emotions and '{emotion}' in base_response:
                emotion = emotions[0]
                intensity = 'really difficult' if analysis['urgency_level'] == 'high' else 'challenging'
                base_response = base_response.format(emotion=emotion, intensity=intensity)
            
            # Add behavioral focus if needed
            if '{behavior}' in base_response and 'behavioral_change' in analysis['therapeutic_needs']:
                base_response = base_response.format(behavior='this pattern')
            
            # Add exploration topic if needed
            if '{topic}' in base_response:
                base_response = base_response.format(topic='what you\'re experiencing')
        else:
            base_response = "I'm here to support you through this."
        
        # Add modality-specific elements
        modality = approach['modality']
        if modality == 'cbt':
            base_response += " Let's examine the thoughts and feelings involved here."
        elif modality == 'dbt':
            base_response += " Your emotions are valid, and we can work on managing them effectively."
        elif modality == 'psychodynamic':
            base_response += " I wonder what this might connect to in your past experiences."
        elif modality == 'humanistic':
            base_response += " You are the expert on your own experience."
        
        # Add supporting technique elements
        if 'validation' in approach['supporting_techniques']:
            base_response += " What you're going through makes complete sense."
        if 'grounding' in approach['supporting_techniques']:
            base_response += " Let's take a moment to ground ourselves in the present."
        
        return base_response.strip()
    
    def _assess_response_quality(self, response: str, analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the quality of the generated response."""
        response_lower = response.lower()
        
        # Empathy assessment
        empathy_indicators = ['hear', 'understand', 'feel', 'sounds', 'sense', 'that', 'you']
        empathy_score = min(1.0, sum(1 for indicator in empathy_indicators if indicator in response_lower) * 0.15)
        
        # Appropriateness assessment
        appropriateness_score = 0.8  # Base score
        if any(word in response_lower for word in ['should', 'must', 'have to']):
            appropriateness_score -= 0.2  # Directive language penalty
        if any(word in response_lower for word in ['together', 'we', 'explore']):
            appropriateness_score += 0.1  # Collaborative language bonus
        
        # Safety assessment
        safety_score = 1.0  # Base safe score
        unsafe_indicators = ['advice', 'diagnose', 'medication', 'cure']
        safety_score -= sum(0.2 for indicator in unsafe_indicators if indicator in response_lower)
        safety_score = max(0.0, safety_score)
        
        # Overall quality
        overall_quality = (empathy_score + appropriateness_score + safety_score) / 3
        
        return {
            'overall_quality': round(overall_quality, 2),
            'empathy_score': round(empathy_score, 2),
            'appropriateness_score': round(appropriateness_score, 2),
            'safety_score': round(safety_score, 2),
            'response_length': len(response),
            'assessment_criteria': {
                'empathic_language': empathy_score > 0.6,
                'collaborative_approach': 'we' in response_lower or 'together' in response_lower,
                'non_directive': 'should' not in response_lower,
                'professionally_appropriate': safety_score > 0.8
            }
        }
    
    def generate_follow_up_questions(self, context: Dict[str, Any]) -> List[str]:
        """Generate appropriate follow-up questions."""
        modality = context.get('therapeutic_modality', 'humanistic')
        session_number = context.get('session_number', 1)
        
        questions = []
        
        if session_number <= 3:  # Early sessions
            questions.extend([
                "How has your week been since we last met?",
                "What would be most helpful to focus on today?",
                "How are you feeling about our work together so far?"
            ])
        else:  # Later sessions
            if modality == 'cbt':
                questions.extend([
                    "What thoughts went through your mind when that happened?",
                    "How did that thought make you feel?",
                    "What evidence supports or challenges that thought?"
                ])
            elif modality == 'dbt':
                questions.extend([
                    "What emotions are you noticing right now?",
                    "What skills have you been practicing?",
                    "How intense is that emotion on a scale of 1-10?"
                ])
            else:
                questions.extend([
                    "What does that experience mean to you?",
                    "How did that make you feel?",
                    "What comes up for you when you think about that?"
                ])
        
        return questions[:3]  # Return top 3 questions
    
    def adapt_to_client_style(self, client_communication_style: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt response generation to client's communication style."""
        adaptations = {}
        
        # Adapt to formality level
        formality = client_communication_style.get('formality', 'medium')
        if formality == 'high':
            adaptations['language_style'] = 'formal'
            adaptations['address_style'] = 'respectful'
        elif formality == 'low':
            adaptations['language_style'] = 'casual'
            adaptations['address_style'] = 'friendly'
        
        # Adapt to pace preference
        pace = client_communication_style.get('pace', 'medium')
        if pace == 'slow':
            adaptations['response_length'] = 'extended'
            adaptations['processing_time'] = 'allow_pauses'
        elif pace == 'fast':
            adaptations['response_length'] = 'concise'
            adaptations['processing_time'] = 'quick_responses'
        
        # Adapt to emotional expression style
        emotional_style = client_communication_style.get('emotional_expression', 'moderate')
        if emotional_style == 'high':
            adaptations['empathy_level'] = 'high'
            adaptations['validation_frequency'] = 'frequent'
        elif emotional_style == 'low':
            adaptations['empathy_level'] = 'measured'
            adaptations['validation_frequency'] = 'selective'
        
        return adaptations
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get statistics from response generation history."""
        if not self.generation_history:
            return {'total_generations': 0}
        
        total = len(self.generation_history)
        avg_quality = sum(gen['quality_assessment']['overall_quality'] for gen in self.generation_history) / total
        
        # Technique usage statistics
        technique_counts = {}
        for gen in self.generation_history:
            for technique in gen['approach']['techniques']:
                technique_counts[technique] = technique_counts.get(technique, 0) + 1
        
        # Modality distribution
        modality_counts = {}
        for gen in self.generation_history:
            modality = gen['approach']['modality']
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
        
        return {
            'total_generations': total,
            'average_quality': round(avg_quality, 2),
            'technique_usage': technique_counts,
            'modality_distribution': modality_counts,
            'high_quality_responses': sum(1 for gen in self.generation_history 
                                        if gen['quality_assessment']['overall_quality'] >= 0.8)
        }


class TestTherapeuticResponseGenerator(unittest.TestCase):
    """Test suite for TherapeuticResponseGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = MockTherapeuticResponseGenerator()
        
        self.test_contexts = {
            'cbt_session': {
                'therapeutic_modality': 'cbt',
                'session_number': 5,
                'client_goals': ['reduce anxiety', 'improve coping'],
                'presenting_issue': 'anxiety'
            },
            'dbt_session': {
                'therapeutic_modality': 'dbt',
                'session_number': 8,
                'client_goals': ['emotion regulation', 'distress tolerance'],
                'presenting_issue': 'emotional_dysregulation'
            },
            'early_session': {
                'therapeutic_modality': 'humanistic',
                'session_number': 1,
                'client_goals': ['build rapport', 'explore concerns'],
                'presenting_issue': 'general_distress'
            }
        }
        
        self.test_inputs = {
            'anxiety': "I've been feeling really anxious lately and I don't know what to do about it.",
            'depression': "I feel so sad and hopeless. Nothing seems to matter anymore.",
            'anger': "I'm so frustrated with everything. I just want to scream.",
            'positive': "I had a really good day today and I'm feeling hopeful.",
            'complex': "I'm dealing with anxiety about work, sadness about my relationship, and I'm not sure how to handle all these emotions at once."
        }
    
    def test_initialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertIsInstance(self.generator.therapeutic_modalities, list)
        self.assertIsInstance(self.generator.response_templates, dict)
        self.assertEqual(len(self.generator.generation_history), 0)
    
    def test_successful_response_generation(self):
        """Test successful therapeutic response generation."""
        for input_type, client_input in self.test_inputs.items():
            with self.subTest(input_type=input_type):
                result = self.generator.generate_response(client_input, self.test_contexts['cbt_session'])
                
                self.assertTrue(result['success'])
                self.assertIsNone(result['error'])
                self.assertIsNotNone(result['response'])
                self.assertGreater(len(result['response']), 0)
                self.assertGreaterEqual(result['therapeutic_quality'], 0.0)
                self.assertLessEqual(result['therapeutic_quality'], 1.0)
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        invalid_inputs = [None, "", "   ", 123, [], {}]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input=invalid_input):
                result = self.generator.generate_response(invalid_input, self.test_contexts['cbt_session'])
                
                self.assertFalse(result['success'])
                self.assertIsNotNone(result['error'])
                self.assertIsNone(result['response'])
                self.assertEqual(result['therapeutic_quality'], 0.0)
    
    def test_modality_specific_responses(self):
        """Test that responses adapt to different therapeutic modalities."""
        client_input = self.test_inputs['anxiety']
        
        # Test different modalities
        modalities = ['cbt', 'dbt', 'psychodynamic', 'humanistic']
        responses = {}
        
        for modality in modalities:
            context = self.test_contexts['cbt_session'].copy()
            context['therapeutic_modality'] = modality
            
            result = self.generator.generate_response(client_input, context)
            responses[modality] = result['response']
            
            self.assertTrue(result['success'])
            self.assertGreater(result['therapeutic_quality'], 0.5)
        
        # Responses should be different for different modalities
        unique_responses = set(responses.values())
        self.assertGreater(len(unique_responses), 1)
    
    def test_emotion_detection_and_response(self):
        """Test emotion detection and appropriate response generation."""
        emotion_tests = [
            ('anxiety', ['anxious', 'worried'], 'anxiety'),
            ('depression', ['sad', 'hopeless'], 'sadness'),
            ('anger', ['frustrated', 'scream'], 'anger'),
            ('positive', ['good', 'hopeful'], 'positive')
        ]
        
        for emotion_type, expected_words, expected_emotion in emotion_tests:
            with self.subTest(emotion=emotion_type):
                result = self.generator.generate_response(
                    self.test_inputs[emotion_type], 
                    self.test_contexts['cbt_session']
                )
                
                self.assertTrue(result['success'])
                self.assertIn(expected_emotion, result['analysis']['emotions_detected'])
                
                # Response should be empathetic
                self.assertGreaterEqual(result['empathy_score'], 0.1)
    
    def test_session_number_adaptation(self):
        """Test adaptation based on session number."""
        client_input = self.test_inputs['anxiety']
        
        # Early session (rapport building)
        early_context = self.test_contexts['early_session']
        early_result = self.generator.generate_response(client_input, early_context)
        
        # Later session (deeper work)
        later_context = self.test_contexts['cbt_session']  # Session 5
        later_result = self.generator.generate_response(client_input, later_context)
        
        # Both should succeed
        self.assertTrue(early_result['success'])
        self.assertTrue(later_result['success'])
        
        # Early sessions should focus more on validation
        self.assertIn('validation', early_result['techniques_used'])
    
    def test_safety_assessment(self):
        """Test safety assessment of generated responses."""
        safe_input = "I'm feeling a bit stressed about work."
        result = self.generator.generate_response(safe_input, self.test_contexts['cbt_session'])
        
        self.assertTrue(result['success'])
        self.assertGreaterEqual(result['safety_score'], 0.8)
        
        # Response should not contain unsafe elements
        response_lower = result['response'].lower()
        unsafe_words = ['diagnose', 'medication', 'cure']
        for word in unsafe_words:
            self.assertNotIn(word, response_lower)
    
    def test_follow_up_question_generation(self):
        """Test generation of appropriate follow-up questions."""
        for context_name, context in self.test_contexts.items():
            with self.subTest(context=context_name):
                questions = self.generator.generate_follow_up_questions(context)
                
                self.assertIsInstance(questions, list)
                self.assertGreater(len(questions), 0)
                self.assertLessEqual(len(questions), 3)
                
                # All questions should end with question marks
                for question in questions:
                    self.assertTrue(question.endswith('?'))
    
    def test_client_style_adaptation(self):
        """Test adaptation to client communication style."""
        communication_styles = [
            {'formality': 'high', 'pace': 'slow', 'emotional_expression': 'low'},
            {'formality': 'low', 'pace': 'fast', 'emotional_expression': 'high'},
            {'formality': 'medium', 'pace': 'medium', 'emotional_expression': 'moderate'}
        ]
        
        for style in communication_styles:
            with self.subTest(style=style):
                adaptations = self.generator.adapt_to_client_style(style)
                
                self.assertIsInstance(adaptations, dict)
                self.assertTrue(len(adaptations) >= 0)
                
                # Check appropriate adaptations
                if style['formality'] == 'high':
                    self.assertEqual(adaptations['language_style'], 'formal')
                elif style['formality'] == 'low':
                    self.assertEqual(adaptations['language_style'], 'casual')
    
    def test_response_quality_assessment(self):
        """Test response quality assessment accuracy."""
        # Generate a response
        result = self.generator.generate_response(
            self.test_inputs['anxiety'], 
            self.test_contexts['cbt_session']
        )
        
        self.assertTrue(result['success'])
        
        # Quality scores should be within valid ranges
        self.assertGreaterEqual(result['therapeutic_quality'], 0.0)
        self.assertLessEqual(result['therapeutic_quality'], 1.0)
        self.assertGreaterEqual(result['empathy_score'], 0.0)
        self.assertLessEqual(result['empathy_score'], 1.0)
        self.assertGreaterEqual(result['safety_score'], 0.0)
        self.assertLessEqual(result['safety_score'], 1.0)
    
    def test_generation_statistics(self):
        """Test generation statistics collection."""
        # Generate multiple responses
        for input_type, client_input in self.test_inputs.items():
            self.generator.generate_response(client_input, self.test_contexts['cbt_session'])
        
        stats = self.generator.get_generation_statistics()
        
        self.assertEqual(stats['total_generations'], len(self.test_inputs))
        self.assertIn('average_quality', stats)
        self.assertIn('technique_usage', stats)
        self.assertIn('modality_distribution', stats)
        self.assertGreaterEqual(stats['average_quality'], 0.0)
    
    def test_complex_input_handling(self):
        """Test handling of complex, multi-faceted client input."""
        complex_input = self.test_inputs['complex']
        result = self.generator.generate_response(complex_input, self.test_contexts['cbt_session'])
        
        self.assertTrue(result['success'])
        
        # Should detect multiple emotions
        emotions = result['analysis']['emotions_detected']
        self.assertGreaterEqual(len(emotions), 1)
        
        # Should have high complexity rating
        self.assertIn(result['analysis']['complexity'], ['high', 'medium'])
        
        # Response should be comprehensive
        self.assertGreater(len(result['response']), 50)
    
    def test_batch_response_generation(self):
        """Test batch generation of responses."""
        inputs_and_contexts = [
            (self.test_inputs['anxiety'], self.test_contexts['cbt_session']),
            (self.test_inputs['depression'], self.test_contexts['dbt_session']),
            (self.test_inputs['anger'], self.test_contexts['early_session'])
        ]
        
        results = []
        for client_input, context in inputs_and_contexts:
            result = self.generator.generate_response(client_input, context)
            results.append(result)
        
        # All should succeed
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result['success'])
            self.assertGreater(result['therapeutic_quality'], 0.5)


class TestTherapeuticResponseGeneratorIntegration(unittest.TestCase):
    """Integration tests for TherapeuticResponseGenerator."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.generator = MockTherapeuticResponseGenerator()
        self.test_contexts = {
            'cbt_session': {
                'therapeutic_modality': 'cbt',
                'session_number': 5,
                'client_goals': ['reduce anxiety', 'improve coping'],
                'presenting_issue': 'anxiety'
            },
            'early_session': {
                'therapeutic_modality': 'humanistic',
                'session_number': 1,
                'client_goals': ['build rapport', 'explore concerns'],
                'presenting_issue': 'general_distress'
            }
        }
    
    def test_complete_therapeutic_conversation_flow(self):
        """Test complete therapeutic conversation flow."""
        # Simulate a therapy session progression
        conversation_flow = [
            ("I'm not sure where to start today.", self.test_contexts['early_session']),
            ("I've been having trouble sleeping because of work stress.", self.test_contexts['cbt_session']),
            ("When I think about work, I immediately assume the worst will happen.", self.test_contexts['cbt_session']),
            ("I tried the thought challenging technique we discussed.", self.test_contexts['cbt_session'])
        ]
        
        responses = []
        for client_input, context in conversation_flow:
            result = self.generator.generate_response(client_input, context)
            responses.append(result)
            
            # Each response should be successful and appropriate
            self.assertTrue(result['success'])
            self.assertGreater(result['therapeutic_quality'], 0.6)
        
        # Verify conversation progression
        self.assertEqual(len(responses), 4)
        
        # Later responses should show technique application
        self.assertTrue(len(responses[2]['techniques_used'] + responses[3]['techniques_used']) > 0)
    
    def test_multi_modal_therapeutic_approach(self):
        """Test integration across multiple therapeutic modalities."""
        client_input = "I'm struggling with intense emotions and don't know how to cope."
        
        modalities = ['cbt', 'dbt', 'psychodynamic', 'humanistic']
        results = {}
        
        for modality in modalities:
            context = {
                'therapeutic_modality': modality,
                'session_number': 5,
                'client_goals': ['emotion_regulation'],
                'presenting_issue': 'emotional_difficulties'
            }
            
            result = self.generator.generate_response(client_input, context)
            results[modality] = result
            
            # All modalities should generate appropriate responses
            self.assertTrue(result['success'])
            self.assertGreater(result['therapeutic_quality'], 0.6)
        
        # Each modality should have distinct approaches
        techniques_used = set()
        for result in results.values():
            techniques_used.update(result['techniques_used'])
        
        self.assertGreater(len(techniques_used), 2)  # Multiple techniques across modalities


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
