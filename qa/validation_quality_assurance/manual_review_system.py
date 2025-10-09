#!/usr/bin/env python3
"""
Manual Review and Validation System - Task 5.7.2.3
Creates comprehensive manual review and validation systems for human oversight of conversation quality.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import uuid
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewStatus(Enum):
    """Review status options"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REQUIRES_REVISION = "requires_revision"
    APPROVED = "approved"
    REJECTED = "rejected"

class ReviewPriority(Enum):
    """Review priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class ReviewerRole(Enum):
    """Reviewer role types"""
    JUNIOR_REVIEWER = "junior_reviewer"
    SENIOR_REVIEWER = "senior_reviewer"
    CLINICAL_SUPERVISOR = "clinical_supervisor"
    QUALITY_MANAGER = "quality_manager"
    SUBJECT_EXPERT = "subject_expert"

@dataclass
class ReviewCriteria:
    """Review criteria definition"""
    criteria_id: str
    name: str
    description: str
    weight: float
    required: bool = True
    scoring_scale: str = "1-5"

@dataclass
class ReviewAssignment:
    """Review assignment details"""
    assignment_id: str
    conversation_id: str
    reviewer_id: str
    reviewer_role: ReviewerRole
    priority: ReviewPriority
    criteria: List[ReviewCriteria]
    assigned_at: str
    due_date: str
    status: ReviewStatus
    estimated_time_minutes: int

@dataclass
class ReviewResult:
    """Individual review result"""
    review_id: str
    assignment_id: str
    conversation_id: str
    reviewer_id: str
    criteria_scores: Dict[str, float]
    overall_score: float
    comments: str
    issues_identified: List[str]
    recommendations: List[str]
    approval_status: str
    review_time_minutes: int
    completed_at: str
    metadata: Dict[str, Any]

class ManualReviewSystem:
    """Enterprise-grade manual review and validation system"""
    
    def __init__(self):
        """Initialize manual review system"""
        self.reviewers = {}
        self.review_criteria = {}
        self.assignments = {}
        self.reviews = {}
        self.review_queue = []
        self.review_stats = {
            'total_assignments': 0,
            'completed_reviews': 0,
            'pending_reviews': 0,
            'average_review_time': 0.0,
            'reviewer_performance': {},
            'quality_trends': []
        }
        self._setup_default_criteria()
        self._setup_default_reviewers()
        
    def _setup_default_criteria(self):
        """Setup default review criteria"""
        
        default_criteria = [
            ReviewCriteria(
                criteria_id="therapeutic_accuracy",
                name="Therapeutic Accuracy",
                description="Accuracy of therapeutic responses and interventions",
                weight=0.25,
                required=True
            ),
            ReviewCriteria(
                criteria_id="clinical_appropriateness",
                name="Clinical Appropriateness",
                description="Appropriateness of clinical language and approach",
                weight=0.20,
                required=True
            ),
            ReviewCriteria(
                criteria_id="safety_compliance",
                name="Safety Compliance",
                description="Adherence to safety protocols and crisis management",
                weight=0.20,
                required=True
            ),
            ReviewCriteria(
                criteria_id="ethical_standards",
                name="Ethical Standards",
                description="Compliance with ethical guidelines and professional boundaries",
                weight=0.15,
                required=True
            ),
            ReviewCriteria(
                criteria_id="communication_quality",
                name="Communication Quality",
                description="Clarity, empathy, and effectiveness of communication",
                weight=0.10,
                required=True
            ),
            ReviewCriteria(
                criteria_id="cultural_sensitivity",
                name="Cultural Sensitivity",
                description="Cultural awareness and inclusive language",
                weight=0.10,
                required=False
            )
        ]
        
        for criteria in default_criteria:
            self.review_criteria[criteria.criteria_id] = criteria
        
        logger.info(f"✅ Setup {len(default_criteria)} default review criteria")
    
    def _setup_default_reviewers(self):
        """Setup default reviewer profiles"""
        
        default_reviewers = [
            {
                'reviewer_id': 'reviewer_001',
                'name': 'Dr. Sarah Johnson',
                'role': ReviewerRole.CLINICAL_SUPERVISOR,
                'specializations': ['anxiety', 'depression', 'trauma'],
                'experience_years': 15,
                'certification': 'Licensed Clinical Psychologist',
                'active': True
            },
            {
                'reviewer_id': 'reviewer_002',
                'name': 'Michael Chen',
                'role': ReviewerRole.SENIOR_REVIEWER,
                'specializations': ['crisis_intervention', 'suicide_prevention'],
                'experience_years': 8,
                'certification': 'Licensed Clinical Social Worker',
                'active': True
            },
            {
                'reviewer_id': 'reviewer_003',
                'name': 'Dr. Maria Rodriguez',
                'role': ReviewerRole.SUBJECT_EXPERT,
                'specializations': ['cultural_competency', 'bilingual_therapy'],
                'experience_years': 12,
                'certification': 'Licensed Marriage and Family Therapist',
                'active': True
            }
        ]
        
        for reviewer in default_reviewers:
            self.reviewers[reviewer['reviewer_id']] = reviewer
            self.review_stats['reviewer_performance'][reviewer['reviewer_id']] = {
                'reviews_completed': 0,
                'average_score_given': 0.0,
                'average_review_time': 0.0,
                'consistency_score': 0.0
            }
        
        logger.info(f"✅ Setup {len(default_reviewers)} default reviewers")
    
    def register_reviewer(self, reviewer_data: Dict[str, Any]) -> str:
        """Register a new reviewer"""
        reviewer_id = reviewer_data.get('reviewer_id', f"reviewer_{uuid.uuid4().hex[:8]}")
        
        required_fields = ['name', 'role', 'specializations', 'experience_years']
        for field in required_fields:
            if field not in reviewer_data:
                raise ValueError(f"Missing required field: {field}")
        
        self.reviewers[reviewer_id] = {
            'reviewer_id': reviewer_id,
            'registered_at': datetime.now().isoformat(),
            'active': True,
            **reviewer_data
        }
        
        self.review_stats['reviewer_performance'][reviewer_id] = {
            'reviews_completed': 0,
            'average_score_given': 0.0,
            'average_review_time': 0.0,
            'consistency_score': 0.0
        }
        
        logger.info(f"✅ Registered reviewer: {reviewer_id}")
        return reviewer_id
    
    def create_review_assignment(self, conversation: Dict[str, Any], 
                               reviewer_id: str = None,
                               priority: ReviewPriority = ReviewPriority.MEDIUM,
                               criteria_ids: List[str] = None,
                               due_hours: int = 24) -> str:
        """Create a new review assignment"""
        
        conversation_id = conversation.get('id', f"conv_{uuid.uuid4().hex[:8]}")
        assignment_id = f"assign_{uuid.uuid4().hex[:8]}"
        
        # Auto-assign reviewer if not specified
        if reviewer_id is None:
            reviewer_id = self._auto_assign_reviewer(conversation, priority)
        
        if reviewer_id not in self.reviewers:
            raise ValueError(f"Reviewer {reviewer_id} not found")
        
        # Use default criteria if not specified
        if criteria_ids is None:
            criteria_ids = list(self.review_criteria.keys())
        
        criteria = [self.review_criteria[cid] for cid in criteria_ids 
                   if cid in self.review_criteria]
        
        # Calculate estimated time based on conversation complexity
        estimated_time = self._estimate_review_time(conversation, criteria)
        
        assignment = ReviewAssignment(
            assignment_id=assignment_id,
            conversation_id=conversation_id,
            reviewer_id=reviewer_id,
            reviewer_role=self.reviewers[reviewer_id]['role'],
            priority=priority,
            criteria=criteria,
            assigned_at=datetime.now().isoformat(),
            due_date=(datetime.now() + timedelta(hours=due_hours)).isoformat(),
            status=ReviewStatus.PENDING,
            estimated_time_minutes=estimated_time
        )
        
        self.assignments[assignment_id] = assignment
        self.review_queue.append(assignment_id)
        self.review_stats['total_assignments'] += 1
        self.review_stats['pending_reviews'] += 1
        
        logger.info(f"✅ Created review assignment {assignment_id} for reviewer {reviewer_id}")
        return assignment_id
    
    def _auto_assign_reviewer(self, conversation: Dict[str, Any], 
                            priority: ReviewPriority) -> str:
        """Automatically assign the best reviewer for a conversation"""
        
        # Extract conversation characteristics
        text = str(conversation.get('conversation', ''))
        
        # Identify required specializations
        specializations_needed = []
        if any(word in text.lower() for word in ['suicide', 'kill', 'harm', 'crisis']):
            specializations_needed.append('crisis_intervention')
        if any(word in text.lower() for word in ['anxiety', 'anxious', 'panic']):
            specializations_needed.append('anxiety')
        if any(word in text.lower() for word in ['depression', 'depressed', 'sad']):
            specializations_needed.append('depression')
        if any(word in text.lower() for word in ['trauma', 'abuse', 'ptsd']):
            specializations_needed.append('trauma')
        
        # Score reviewers based on availability and expertise
        reviewer_scores = {}
        for reviewer_id, reviewer in self.reviewers.items():
            if not reviewer.get('active', True):
                continue
            
            score = 0.0
            
            # Experience weight
            score += min(reviewer.get('experience_years', 0) / 20.0, 1.0) * 0.3
            
            # Specialization match
            reviewer_specs = reviewer.get('specializations', [])
            matching_specs = len(set(specializations_needed) & set(reviewer_specs))
            if specializations_needed:
                score += (matching_specs / len(specializations_needed)) * 0.4
            else:
                score += 0.4  # No specific specialization needed
            
            # Role appropriateness
            role_weights = {
                ReviewerRole.CLINICAL_SUPERVISOR: 1.0,
                ReviewerRole.SENIOR_REVIEWER: 0.8,
                ReviewerRole.SUBJECT_EXPERT: 0.9,
                ReviewerRole.QUALITY_MANAGER: 0.7,
                ReviewerRole.JUNIOR_REVIEWER: 0.5
            }
            score += role_weights.get(reviewer['role'], 0.5) * 0.2
            
            # Current workload (lower is better)
            current_assignments = sum(1 for a in self.assignments.values() 
                                    if a.reviewer_id == reviewer_id and 
                                    a.status in [ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS])
            workload_penalty = min(current_assignments * 0.1, 0.5)
            score -= workload_penalty * 0.1
            
            reviewer_scores[reviewer_id] = score
        
        # Select best reviewer
        if not reviewer_scores:
            raise ValueError("No available reviewers found")
        
        best_reviewer = max(reviewer_scores.items(), key=lambda x: x[1])[0]
        return best_reviewer
    
    def _estimate_review_time(self, conversation: Dict[str, Any], 
                            criteria: List[ReviewCriteria]) -> int:
        """Estimate review time in minutes"""
        
        # Base time
        base_time = 15  # minutes
        
        # Conversation complexity
        text_length = len(str(conversation.get('conversation', '')))
        complexity_time = min(text_length / 1000 * 5, 30)  # 5 min per 1000 chars, max 30
        
        # Criteria complexity
        criteria_time = len(criteria) * 3  # 3 minutes per criteria
        
        # Safety content requires more time
        text = str(conversation.get('conversation', '')).lower()
        if any(word in text for word in ['suicide', 'crisis', 'harm', 'abuse']):
            safety_time = 15
        else:
            safety_time = 0
        
        total_time = base_time + complexity_time + criteria_time + safety_time
        return int(total_time)
    
    def submit_review(self, assignment_id: str, reviewer_id: str,
                     criteria_scores: Dict[str, float],
                     comments: str,
                     issues_identified: List[str] = None,
                     recommendations: List[str] = None,
                     approval_status: str = "approved") -> str:
        """Submit a completed review"""
        
        if assignment_id not in self.assignments:
            raise ValueError(f"Assignment {assignment_id} not found")
        
        assignment = self.assignments[assignment_id]
        
        if assignment.reviewer_id != reviewer_id:
            raise ValueError(f"Reviewer {reviewer_id} not assigned to this review")
        
        if assignment.status != ReviewStatus.PENDING:
            raise ValueError(f"Assignment {assignment_id} is not pending review")
        
        # Validate criteria scores
        for criteria in assignment.criteria:
            if criteria.criteria_id not in criteria_scores:
                if criteria.required:
                    raise ValueError(f"Missing required criteria score: {criteria.criteria_id}")
                else:
                    criteria_scores[criteria.criteria_id] = 3.0  # Default neutral score
        
        # Calculate overall score
        total_weight = sum(c.weight for c in assignment.criteria)
        overall_score = sum(
            criteria_scores.get(c.criteria_id, 3.0) * c.weight 
            for c in assignment.criteria
        ) / total_weight
        
        # Create review result
        review_id = f"review_{uuid.uuid4().hex[:8]}"
        
        review_result = ReviewResult(
            review_id=review_id,
            assignment_id=assignment_id,
            conversation_id=assignment.conversation_id,
            reviewer_id=reviewer_id,
            criteria_scores=criteria_scores,
            overall_score=overall_score,
            comments=comments,
            issues_identified=issues_identified or [],
            recommendations=recommendations or [],
            approval_status=approval_status,
            review_time_minutes=assignment.estimated_time_minutes,  # Would be actual time in real system
            completed_at=datetime.now().isoformat(),
            metadata={}
        )
        
        self.reviews[review_id] = review_result
        
        # Update assignment status
        assignment.status = ReviewStatus.COMPLETED
        
        # Update statistics
        self._update_review_stats(review_result)
        
        # Remove from queue
        if assignment_id in self.review_queue:
            self.review_queue.remove(assignment_id)
        
        logger.info(f"✅ Review {review_id} submitted for assignment {assignment_id}")
        return review_id
    
    def _update_review_stats(self, review: ReviewResult):
        """Update review statistics"""
        self.review_stats['completed_reviews'] += 1
        self.review_stats['pending_reviews'] = max(0, self.review_stats['pending_reviews'] - 1)
        
        # Update reviewer performance
        reviewer_id = review.reviewer_id
        if reviewer_id in self.review_stats['reviewer_performance']:
            perf = self.review_stats['reviewer_performance'][reviewer_id]
            
            # Update completed count
            perf['reviews_completed'] += 1
            
            # Update average score given
            total_reviews = perf['reviews_completed']
            current_avg = perf['average_score_given']
            perf['average_score_given'] = ((current_avg * (total_reviews - 1)) + review.overall_score) / total_reviews
            
            # Update average review time
            current_time_avg = perf['average_review_time']
            perf['average_review_time'] = ((current_time_avg * (total_reviews - 1)) + review.review_time_minutes) / total_reviews
        
        # Update overall average review time
        total_completed = self.review_stats['completed_reviews']
        current_avg_time = self.review_stats['average_review_time']
        self.review_stats['average_review_time'] = (
            (current_avg_time * (total_completed - 1) + review.review_time_minutes) / total_completed
        )
        
        # Add to quality trends
        self.review_stats['quality_trends'].append({
            'timestamp': review.completed_at,
            'overall_score': review.overall_score,
            'reviewer_id': review.reviewer_id
        })
    
    def get_pending_assignments(self, reviewer_id: str = None) -> List[ReviewAssignment]:
        """Get pending review assignments"""
        pending = []
        
        for assignment_id in self.review_queue:
            if assignment_id in self.assignments:
                assignment = self.assignments[assignment_id]
                if assignment.status == ReviewStatus.PENDING:
                    if reviewer_id is None or assignment.reviewer_id == reviewer_id:
                        pending.append(assignment)
        
        # Sort by priority and due date
        priority_order = {
            ReviewPriority.URGENT: 0,
            ReviewPriority.HIGH: 1,
            ReviewPriority.MEDIUM: 2,
            ReviewPriority.LOW: 3
        }
        
        pending.sort(key=lambda x: (priority_order[x.priority], x.due_date))
        return pending
    
    def get_review_results(self, conversation_id: str = None, 
                          reviewer_id: str = None) -> List[ReviewResult]:
        """Get review results with optional filtering"""
        results = list(self.reviews.values())
        
        if conversation_id:
            results = [r for r in results if r.conversation_id == conversation_id]
        
        if reviewer_id:
            results = [r for r in results if r.reviewer_id == reviewer_id]
        
        return results
    
    def generate_review_report(self, start_date: str = None, 
                             end_date: str = None) -> Dict[str, Any]:
        """Generate comprehensive review report"""
        
        # Filter reviews by date range if specified
        reviews = list(self.reviews.values())
        if start_date or end_date:
            filtered_reviews = []
            for review in reviews:
                review_date = datetime.fromisoformat(review.completed_at.replace('Z', '+00:00'))
                
                if start_date:
                    start = datetime.fromisoformat(start_date)
                    if review_date < start:
                        continue
                
                if end_date:
                    end = datetime.fromisoformat(end_date)
                    if review_date > end:
                        continue
                
                filtered_reviews.append(review)
            
            reviews = filtered_reviews
        
        if not reviews:
            return {'error': 'No reviews found for specified criteria'}
        
        # Calculate report metrics
        total_reviews = len(reviews)
        average_score = sum(r.overall_score for r in reviews) / total_reviews
        
        # Score distribution
        score_ranges = {'1-2': 0, '2-3': 0, '3-4': 0, '4-5': 0}
        for review in reviews:
            score = review.overall_score
            if score < 2:
                score_ranges['1-2'] += 1
            elif score < 3:
                score_ranges['2-3'] += 1
            elif score < 4:
                score_ranges['3-4'] += 1
            else:
                score_ranges['4-5'] += 1
        
        # Reviewer performance
        reviewer_performance = {}
        for review in reviews:
            rid = review.reviewer_id
            if rid not in reviewer_performance:
                reviewer_performance[rid] = {
                    'reviews_completed': 0,
                    'average_score': 0.0,
                    'total_score': 0.0
                }
            
            reviewer_performance[rid]['reviews_completed'] += 1
            reviewer_performance[rid]['total_score'] += review.overall_score
        
        for rid, perf in reviewer_performance.items():
            perf['average_score'] = perf['total_score'] / perf['reviews_completed']
            del perf['total_score']
        
        # Common issues
        all_issues = []
        for review in reviews:
            all_issues.extend(review.issues_identified)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'report_period': {
                'start_date': start_date,
                'end_date': end_date,
                'generated_at': datetime.now().isoformat()
            },
            'summary_metrics': {
                'total_reviews': total_reviews,
                'average_score': round(average_score, 3),
                'score_distribution': score_ranges
            },
            'reviewer_performance': reviewer_performance,
            'common_issues': common_issues,
            'quality_trends': self.review_stats['quality_trends'][-50:],  # Last 50 reviews
            'system_statistics': self.get_review_statistics()
        }
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """Get comprehensive review system statistics"""
        return {
            'review_stats': self.review_stats,
            'active_reviewers': len([r for r in self.reviewers.values() if r.get('active', True)]),
            'total_reviewers': len(self.reviewers),
            'review_criteria_count': len(self.review_criteria),
            'pending_assignments': len(self.review_queue),
            'statistics_timestamp': datetime.now().isoformat()
        }
    
    def export_review_data(self, output_path: str, include_sensitive: bool = False) -> bool:
        """Export review data to JSON file"""
        try:
            export_data = {
                'assignments': [asdict(a) for a in self.assignments.values()],
                'reviews': [asdict(r) for r in self.reviews.values()],
                'review_criteria': [asdict(c) for c in self.review_criteria.values()],
                'system_statistics': self.get_review_statistics(),
                'export_timestamp': datetime.now().isoformat()
            }
            
            if include_sensitive:
                export_data['reviewers'] = self.reviewers
            else:
                # Export reviewer data without sensitive information
                export_data['reviewers'] = {
                    rid: {k: v for k, v in reviewer.items() 
                         if k not in ['email', 'phone', 'address']}
                    for rid, reviewer in self.reviewers.items()
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Exported review data to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting review data: {e}")
            return False

def main():
    """Test manual review system"""
    review_system = ManualReviewSystem()
    
    # Test conversation
    test_conversation = {
        'id': 'test_manual_001',
        'conversation': 'User: I feel really anxious about my job interview tomorrow. Assistant: I understand job interviews can be anxiety-provoking. Let\'s work on some coping strategies. First, try some deep breathing exercises...'
    }
    
    # Create review assignment
    assignment_id = review_system.create_review_assignment(
        conversation=test_conversation,
        priority=ReviewPriority.HIGH
    )
    
    print(f"✅ Created assignment: {assignment_id}")
    
    # Get pending assignments
    pending = review_system.get_pending_assignments()
    print(f"📋 Pending assignments: {len(pending)}")
    
    if pending:
        assignment = pending[0]
        print(f"Assignment details:")
        print(f"  - Reviewer: {assignment.reviewer_id}")
        print(f"  - Priority: {assignment.priority.value}")
        print(f"  - Estimated time: {assignment.estimated_time_minutes} minutes")
        print(f"  - Criteria count: {len(assignment.criteria)}")
        
        # Submit a test review
        criteria_scores = {
            'therapeutic_accuracy': 4.0,
            'clinical_appropriateness': 4.5,
            'safety_compliance': 5.0,
            'ethical_standards': 4.0,
            'communication_quality': 4.5,
            'cultural_sensitivity': 4.0
        }
        
        review_id = review_system.submit_review(
            assignment_id=assignment.assignment_id,
            reviewer_id=assignment.reviewer_id,
            criteria_scores=criteria_scores,
            comments="Good therapeutic response with appropriate coping strategies. Clear communication and empathetic tone.",
            issues_identified=[],
            recommendations=["Consider adding more specific grounding techniques"],
            approval_status="approved"
        )
        
        print(f"✅ Submitted review: {review_id}")
        
        # Generate report
        report = review_system.generate_review_report()
        print(f"\n📊 Review Report:")
        print(f"Total reviews: {report['summary_metrics']['total_reviews']}")
        print(f"Average score: {report['summary_metrics']['average_score']}")
        print(f"Score distribution: {report['summary_metrics']['score_distribution']}")

if __name__ == "__main__":
    main()
