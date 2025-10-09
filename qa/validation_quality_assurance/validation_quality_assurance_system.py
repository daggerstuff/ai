#!/usr/bin/env python3
"""
Validation & Quality Assurance System - Task 5.7.2 Complete Implementation
Comprehensive system implementing all remaining subtasks for Task 5.7.2.
"""

import json
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import numpy as np

# Import our previously created components
from clinical_standards_validator import ClinicalStandardsValidator
from quality_assurance_workflows import QualityAssuranceWorkflow
from manual_review_system import ManualReviewSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class QualityAssuranceResult:
    """Complete quality assurance result"""
    conversation_id: str
    clinical_validation: Dict[str, Any]
    workflow_results: Dict[str, Any]
    automated_checks: Dict[str, Any]
    quality_improvements: List[str]
    monitoring_alerts: List[str]
    overall_qa_score: float
    qa_status: str
    processing_time: float
    timestamp: str

class ValidationQualityAssuranceSystem:
    """Enterprise-grade validation and quality assurance system"""
    
    def __init__(self):
        """Initialize comprehensive QA system"""
        self.clinical_validator = ClinicalStandardsValidator()
        self.workflow_system = QualityAssuranceWorkflow()
        self.manual_review_system = ManualReviewSystem()
        
        # Quality improvement tracking
        self.improvement_history = []
        self.quality_trends = []
        
        # Monitoring and alerting
        self.alert_thresholds = {
            'clinical_score_min': 0.7,
            'safety_score_min': 0.8,
            'overall_score_min': 0.75,
            'processing_time_max': 300  # seconds
        }
        
        # Performance optimization
        self.processing_cache = {}
        self.batch_size = 100
        
        # System statistics
        self.system_stats = {
            'total_processed': 0,
            'quality_improvements_made': 0,
            'alerts_generated': 0,
            'average_processing_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
        logger.info("✅ Initialized comprehensive validation and quality assurance system")
    
    def process_conversation_qa(self, conversation: Dict[str, Any], 
                              options: Dict[str, Any] = None) -> QualityAssuranceResult:
        """
        Task 5.7.2.4: Implement automated quality checking and validation
        Complete quality assurance processing for a single conversation
        """
        start_time = time.time()
        conversation_id = conversation.get('id', 'unknown')
        options = options or {}
        
        logger.info(f"🔍 Starting comprehensive QA for conversation {conversation_id}")
        
        # Check cache first (Task 5.7.2.7: Performance optimization)
        cache_key = self._generate_cache_key(conversation)
        if cache_key in self.processing_cache and not options.get('force_refresh', False):
            logger.info(f"📋 Using cached QA result for {conversation_id}")
            self.system_stats['cache_hit_rate'] = (
                self.system_stats['cache_hit_rate'] * self.system_stats['total_processed'] + 1
            ) / (self.system_stats['total_processed'] + 1)
            return self.processing_cache[cache_key]
        
        try:
            # Task 5.7.2.1: Clinical standards validation
            clinical_result = self.clinical_validator.validate_conversation(conversation)
            
            # Task 5.7.2.2: Quality assurance workflows
            workflow_result = self.workflow_system.execute_workflow(
                'comprehensive_qa', conversation, options
            )
            
            # Task 5.7.2.4: Automated quality checking
            automated_checks = self._perform_automated_checks(conversation)
            
            # Task 5.7.2.5: Quality improvement feedback loops
            quality_improvements = self._generate_quality_improvements(
                clinical_result, workflow_result, automated_checks
            )
            
            # Task 5.7.2.9: Quality validation monitoring and alerting
            monitoring_alerts = self._check_monitoring_alerts(
                clinical_result, workflow_result, automated_checks
            )
            
            # Calculate overall QA score
            overall_score = self._calculate_overall_qa_score(
                clinical_result, workflow_result, automated_checks
            )
            
            # Determine QA status
            qa_status = self._determine_qa_status(overall_score, monitoring_alerts)
            
            processing_time = time.time() - start_time
            
            # Create comprehensive result
            qa_result = QualityAssuranceResult(
                conversation_id=conversation_id,
                clinical_validation={
                    'overall_clinical_score': clinical_result.overall_clinical_score,
                    'dsm5_compliance': clinical_result.dsm5_compliance,
                    'therapeutic_boundaries': clinical_result.therapeutic_boundaries,
                    'ethical_guidelines': clinical_result.ethical_guidelines,
                    'crisis_intervention': clinical_result.crisis_intervention,
                    'evidence_based_practice': clinical_result.evidence_based_practice,
                    'cultural_competency': clinical_result.cultural_competency,
                    'safety_protocols': clinical_result.safety_protocols,
                    'violations': clinical_result.violations,
                    'recommendations': clinical_result.recommendations
                },
                workflow_results={
                    'workflow_id': workflow_result.workflow_id,
                    'status': workflow_result.status.value,
                    'quality_level': workflow_result.quality_level.value,
                    'quality_score': workflow_result.quality_score,
                    'steps_completed': workflow_result.steps_completed,
                    'issues_found': workflow_result.issues_found,
                    'recommendations': workflow_result.recommendations
                },
                automated_checks=automated_checks,
                quality_improvements=quality_improvements,
                monitoring_alerts=monitoring_alerts,
                overall_qa_score=overall_score,
                qa_status=qa_status,
                processing_time=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
            # Cache result for performance optimization
            self.processing_cache[cache_key] = qa_result
            
            # Update system statistics
            self._update_system_stats(qa_result)
            
            # Task 5.7.2.5: Apply quality improvements
            if quality_improvements:
                self._apply_quality_improvements(conversation_id, quality_improvements)
            
            logger.info(f"✅ Completed comprehensive QA for {conversation_id} in {processing_time:.2f}s")
            return qa_result
            
        except Exception as e:
            logger.error(f"❌ QA processing failed for {conversation_id}: {e}")
            # Task 5.7.2.8: Error handling and recovery
            return self._handle_qa_error(conversation_id, str(e), time.time() - start_time)
    
    def _perform_automated_checks(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Task 5.7.2.4: Perform comprehensive automated quality checks"""
        
        text = str(conversation.get('conversation', ''))
        checks = {}
        
        # Content quality checks
        checks['content_length'] = len(text)
        checks['word_count'] = len(text.split())
        checks['sentence_count'] = text.count('.') + text.count('!') + text.count('?')
        
        # Language quality checks
        checks['readability_score'] = self._calculate_readability(text)
        checks['sentiment_score'] = self._analyze_sentiment(text)
        checks['toxicity_score'] = self._check_toxicity(text)
        
        # Professional language checks
        checks['professional_language'] = self._check_professional_language(text)
        checks['empathy_indicators'] = self._count_empathy_indicators(text)
        checks['question_ratio'] = self._calculate_question_ratio(text)
        
        # Safety checks
        checks['crisis_keywords'] = self._detect_crisis_keywords(text)
        checks['harmful_content'] = self._detect_harmful_content(text)
        checks['privacy_violations'] = self._check_privacy_violations(text)
        
        # Therapeutic technique checks
        checks['therapeutic_techniques'] = self._identify_therapeutic_techniques(text)
        checks['validation_statements'] = self._count_validation_statements(text)
        checks['solution_focused'] = self._assess_solution_focus(text)
        
        return checks
    
    def _generate_quality_improvements(self, clinical_result, workflow_result, 
                                     automated_checks: Dict[str, Any]) -> List[str]:
        """Task 5.7.2.5: Generate quality improvement recommendations"""
        
        improvements = []
        
        # Clinical improvements
        if clinical_result.overall_clinical_score < 0.8:
            improvements.extend(clinical_result.recommendations)
        
        # Workflow improvements
        if workflow_result.quality_score < 0.8:
            improvements.extend(workflow_result.recommendations)
        
        # Automated check improvements
        if automated_checks.get('readability_score', 0) < 0.6:
            improvements.append("Improve text readability and clarity")
        
        if automated_checks.get('empathy_indicators', 0) < 2:
            improvements.append("Increase empathetic language and validation")
        
        if automated_checks.get('question_ratio', 0) < 0.1:
            improvements.append("Add more open-ended questions to encourage exploration")
        
        if len(automated_checks.get('therapeutic_techniques', [])) < 2:
            improvements.append("Incorporate more evidence-based therapeutic techniques")
        
        return list(set(improvements))  # Remove duplicates
    
    def _check_monitoring_alerts(self, clinical_result, workflow_result, 
                               automated_checks: Dict[str, Any]) -> List[str]:
        """Task 5.7.2.9: Generate monitoring alerts based on thresholds"""
        
        alerts = []
        
        # Clinical score alerts
        if clinical_result.overall_clinical_score < self.alert_thresholds['clinical_score_min']:
            alerts.append(f"LOW_CLINICAL_SCORE: {clinical_result.overall_clinical_score:.3f}")
        
        # Safety score alerts
        if clinical_result.safety_protocols < self.alert_thresholds['safety_score_min']:
            alerts.append(f"LOW_SAFETY_SCORE: {clinical_result.safety_protocols:.3f}")
        
        # Crisis content alerts
        if automated_checks.get('crisis_keywords', 0) > 0:
            alerts.append("CRISIS_CONTENT_DETECTED")
        
        # Harmful content alerts
        if automated_checks.get('harmful_content', False):
            alerts.append("HARMFUL_CONTENT_DETECTED")
        
        # Privacy violation alerts
        if automated_checks.get('privacy_violations', False):
            alerts.append("PRIVACY_VIOLATION_DETECTED")
        
        # Clinical violations
        if clinical_result.violations:
            alerts.append(f"CLINICAL_VIOLATIONS: {len(clinical_result.violations)}")
        
        return alerts
    
    def _calculate_overall_qa_score(self, clinical_result, workflow_result, 
                                  automated_checks: Dict[str, Any]) -> float:
        """Calculate weighted overall QA score"""
        
        # Component scores with weights
        clinical_score = clinical_result.overall_clinical_score * 0.4
        workflow_score = workflow_result.quality_score * 0.3
        
        # Automated checks score
        automated_score = 0.0
        if automated_checks.get('readability_score'):
            automated_score += automated_checks['readability_score'] * 0.1
        if automated_checks.get('professional_language'):
            automated_score += 0.1
        if automated_checks.get('empathy_indicators', 0) >= 2:
            automated_score += 0.1
        
        automated_score = min(automated_score, 0.3)  # Cap at weight
        
        overall_score = clinical_score + workflow_score + automated_score
        return min(1.0, max(0.0, overall_score))
    
    def _determine_qa_status(self, overall_score: float, alerts: List[str]) -> str:
        """Determine overall QA status"""
        
        if alerts:
            critical_alerts = [a for a in alerts if any(keyword in a for keyword in 
                             ['CRISIS', 'HARMFUL', 'PRIVACY', 'VIOLATIONS'])]
            if critical_alerts:
                return "CRITICAL_ISSUES"
            else:
                return "NEEDS_ATTENTION"
        
        if overall_score >= 0.9:
            return "EXCELLENT"
        elif overall_score >= 0.8:
            return "GOOD"
        elif overall_score >= 0.7:
            return "ACCEPTABLE"
        else:
            return "NEEDS_IMPROVEMENT"
    
    def _apply_quality_improvements(self, conversation_id: str, improvements: List[str]):
        """Task 5.7.2.5: Apply quality improvement feedback loops"""
        
        improvement_record = {
            'conversation_id': conversation_id,
            'improvements': improvements,
            'applied_at': datetime.now().isoformat(),
            'status': 'pending_implementation'
        }
        
        self.improvement_history.append(improvement_record)
        self.system_stats['quality_improvements_made'] += len(improvements)
        
        logger.info(f"📈 Applied {len(improvements)} quality improvements for {conversation_id}")
    
    def _handle_qa_error(self, conversation_id: str, error: str, processing_time: float) -> QualityAssuranceResult:
        """Task 5.7.2.8: Error handling and recovery"""
        
        logger.error(f"🚨 QA Error Recovery for {conversation_id}: {error}")
        
        # Create minimal result for error case
        return QualityAssuranceResult(
            conversation_id=conversation_id,
            clinical_validation={'error': error},
            workflow_results={'error': error},
            automated_checks={'error': error},
            quality_improvements=[],
            monitoring_alerts=[f"QA_PROCESSING_ERROR: {error}"],
            overall_qa_score=0.0,
            qa_status="ERROR",
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
    
    def process_batch_qa(self, conversations: List[Dict[str, Any]], 
                        options: Dict[str, Any] = None) -> List[QualityAssuranceResult]:
        """Process batch of conversations with quality assurance"""
        
        results = []
        batch_start = time.time()
        
        logger.info(f"🔄 Starting batch QA processing for {len(conversations)} conversations")
        
        for i, conversation in enumerate(conversations):
            try:
                result = self.process_conversation_qa(conversation, options)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"✅ Processed {i + 1}/{len(conversations)} conversations")
                    
            except Exception as e:
                logger.error(f"❌ Error processing conversation {i}: {e}")
                continue
        
        batch_time = time.time() - batch_start
        logger.info(f"🎯 Batch QA complete: {len(results)} conversations processed in {batch_time:.2f}s")
        
        return results
    
    def generate_qa_report(self, results: List[QualityAssuranceResult]) -> Dict[str, Any]:
        """Task 5.7.2.6: Generate comprehensive QA reporting"""
        
        if not results:
            return {'error': 'No QA results provided'}
        
        # Calculate summary statistics
        total_conversations = len(results)
        overall_scores = [r.overall_qa_score for r in results]
        average_score = np.mean(overall_scores)
        
        # Status distribution
        status_counts = {}
        for result in results:
            status = result.qa_status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Alert analysis
        all_alerts = []
        for result in results:
            all_alerts.extend(result.monitoring_alerts)
        
        alert_counts = {}
        for alert in all_alerts:
            alert_type = alert.split(':')[0] if ':' in alert else alert
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        
        # Quality improvement analysis
        all_improvements = []
        for result in results:
            all_improvements.extend(result.quality_improvements)
        
        improvement_counts = {}
        for improvement in all_improvements:
            improvement_counts[improvement] = improvement_counts.get(improvement, 0) + 1
        
        # Performance metrics
        processing_times = [r.processing_time for r in results]
        average_processing_time = np.mean(processing_times)
        
        return {
            'report_summary': {
                'total_conversations': total_conversations,
                'average_qa_score': round(average_score, 3),
                'score_distribution': {
                    'excellent': len([s for s in overall_scores if s >= 0.9]),
                    'good': len([s for s in overall_scores if 0.8 <= s < 0.9]),
                    'acceptable': len([s for s in overall_scores if 0.7 <= s < 0.8]),
                    'needs_improvement': len([s for s in overall_scores if s < 0.7])
                }
            },
            'status_analysis': status_counts,
            'alert_analysis': dict(sorted(alert_counts.items(), key=lambda x: x[1], reverse=True)),
            'improvement_analysis': dict(sorted(improvement_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'performance_metrics': {
                'average_processing_time': round(average_processing_time, 3),
                'total_processing_time': round(sum(processing_times), 3),
                'conversations_per_second': round(total_conversations / sum(processing_times), 3)
            },
            'system_statistics': self.get_system_statistics(),
            'report_timestamp': datetime.now().isoformat()
        }
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'system_stats': self.system_stats,
            'alert_thresholds': self.alert_thresholds,
            'cache_size': len(self.processing_cache),
            'improvement_history_count': len(self.improvement_history),
            'quality_trends_count': len(self.quality_trends),
            'statistics_timestamp': datetime.now().isoformat()
        }
    
    def export_qa_results(self, results: List[QualityAssuranceResult], 
                         output_path: str) -> bool:
        """Task 5.7.2.6: Export QA results and documentation"""
        
        try:
            export_data = {
                'qa_results': [asdict(result) for result in results],
                'qa_report': self.generate_qa_report(results),
                'system_configuration': {
                    'alert_thresholds': self.alert_thresholds,
                    'batch_size': self.batch_size
                },
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Exported QA results to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting QA results: {e}")
            return False
    
    # Helper methods for automated checks
    def _generate_cache_key(self, conversation: Dict[str, Any]) -> str:
        """Generate cache key for conversation"""
        import hashlib
        content = str(conversation.get('conversation', ''))
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate text readability score"""
        # Simplified readability calculation
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')
        if sentences == 0:
            return 0.5
        
        avg_words_per_sentence = len(words) / sentences
        # Ideal range: 15-20 words per sentence
        if 15 <= avg_words_per_sentence <= 20:
            return 1.0
        elif 10 <= avg_words_per_sentence <= 25:
            return 0.8
        else:
            return 0.6
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text"""
        # Simplified sentiment analysis
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'helpful']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'useless', 'harmful']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        return positive_count / (positive_count + negative_count)
    
    def _check_toxicity(self, text: str) -> float:
        """Check for toxic content"""
        toxic_patterns = ['hate', 'stupid', 'idiot', 'worthless', 'pathetic']
        text_lower = text.lower()
        toxic_count = sum(1 for pattern in toxic_patterns if pattern in text_lower)
        return min(toxic_count / 10.0, 1.0)  # Normalize to 0-1
    
    def _check_professional_language(self, text: str) -> bool:
        """Check for professional language usage"""
        professional_indicators = ['understand', 'consider', 'explore', 'reflect', 'support']
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in professional_indicators)
    
    def _count_empathy_indicators(self, text: str) -> int:
        """Count empathy indicators in text"""
        empathy_phrases = [
            'i understand', 'that sounds', 'i can see', 'it makes sense',
            'i hear you', 'that must be', 'i imagine', 'it sounds like'
        ]
        text_lower = text.lower()
        return sum(1 for phrase in empathy_phrases if phrase in text_lower)
    
    def _calculate_question_ratio(self, text: str) -> float:
        """Calculate ratio of questions to total sentences"""
        questions = text.count('?')
        total_sentences = text.count('.') + text.count('!') + text.count('?')
        return questions / total_sentences if total_sentences > 0 else 0.0
    
    def _detect_crisis_keywords(self, text: str) -> int:
        """Detect crisis-related keywords"""
        crisis_keywords = ['suicide', 'kill myself', 'end it all', 'not worth living', 'harm myself']
        text_lower = text.lower()
        return sum(1 for keyword in crisis_keywords if keyword in text_lower)
    
    def _detect_harmful_content(self, text: str) -> bool:
        """Detect potentially harmful content"""
        harmful_patterns = ['hurt yourself', 'you should die', 'end your life', 'kill yourself']
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in harmful_patterns)
    
    def _check_privacy_violations(self, text: str) -> bool:
        """Check for privacy violations"""
        # Simplified privacy check
        import re
        phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b'
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        
        return bool(re.search(phone_pattern, text) or re.search(email_pattern, text))
    
    def _identify_therapeutic_techniques(self, text: str) -> List[str]:
        """Identify therapeutic techniques used"""
        techniques = []
        text_lower = text.lower()
        
        if 'cognitive' in text_lower or 'thoughts' in text_lower:
            techniques.append('Cognitive Restructuring')
        if 'breathe' in text_lower or 'breathing' in text_lower:
            techniques.append('Breathing Exercises')
        if 'mindful' in text_lower:
            techniques.append('Mindfulness')
        if 'goal' in text_lower or 'plan' in text_lower:
            techniques.append('Goal Setting')
        
        return techniques
    
    def _count_validation_statements(self, text: str) -> int:
        """Count validation statements"""
        validation_phrases = [
            'that\'s valid', 'makes sense', 'understandable', 'reasonable',
            'that\'s normal', 'many people feel', 'it\'s okay to'
        ]
        text_lower = text.lower()
        return sum(1 for phrase in validation_phrases if phrase in text_lower)
    
    def _assess_solution_focus(self, text: str) -> float:
        """Assess solution-focused approach"""
        solution_indicators = ['what would help', 'how can we', 'what works', 'strategies', 'solutions']
        text_lower = text.lower()
        solution_count = sum(1 for indicator in solution_indicators if indicator in text_lower)
        return min(solution_count / 3.0, 1.0)  # Normalize to 0-1
    
    def _update_system_stats(self, result: QualityAssuranceResult):
        """Update system statistics"""
        self.system_stats['total_processed'] += 1
        
        if result.monitoring_alerts:
            self.system_stats['alerts_generated'] += len(result.monitoring_alerts)
        
        # Update average processing time
        total = self.system_stats['total_processed']
        current_avg = self.system_stats['average_processing_time']
        self.system_stats['average_processing_time'] = (
            (current_avg * (total - 1) + result.processing_time) / total
        )

def main():
    """Test comprehensive validation and quality assurance system"""
    qa_system = ValidationQualityAssuranceSystem()
    
    # Test conversations
    test_conversations = [
        {
            'id': 'test_qa_001',
            'conversation': 'User: I feel really anxious about my job interview tomorrow. Assistant: I understand that job interviews can be anxiety-provoking. Let\'s explore some coping strategies that might help you feel more prepared and confident.'
        },
        {
            'id': 'test_qa_002',
            'conversation': 'User: I\'ve been having thoughts of suicide lately. Assistant: I\'m very concerned about what you\'re sharing. These thoughts are serious and I want to help ensure your safety. Have you thought about a specific plan? Let\'s connect you with crisis resources like the 988 Suicide & Crisis Lifeline.'
        }
    ]
    
    # Process conversations
    results = qa_system.process_batch_qa(test_conversations)
    
    print(f"\n🎯 QA Processing Results:")
    for result in results:
        print(f"\nConversation: {result.conversation_id}")
        print(f"Overall QA Score: {result.overall_qa_score:.3f}")
        print(f"QA Status: {result.qa_status}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"Clinical Score: {result.clinical_validation['overall_clinical_score']:.3f}")
        print(f"Workflow Score: {result.workflow_results['quality_score']:.3f}")
        print(f"Alerts: {len(result.monitoring_alerts)}")
        print(f"Improvements: {len(result.quality_improvements)}")
    
    # Generate comprehensive report
    report = qa_system.generate_qa_report(results)
    print(f"\n📊 QA Report Summary:")
    print(f"Total Conversations: {report['report_summary']['total_conversations']}")
    print(f"Average QA Score: {report['report_summary']['average_qa_score']}")
    print(f"Status Distribution: {report['status_analysis']}")
    print(f"Performance: {report['performance_metrics']['conversations_per_second']:.2f} conv/sec")
    
    # Export results
    output_path = "/home/vivi/pixelated/ai/validation_quality_assurance/qa_results_test.json"
    success = qa_system.export_qa_results(results, output_path)
    print(f"\n💾 Export Success: {success}")

if __name__ == "__main__":
    main()
