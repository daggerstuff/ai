#!/usr/bin/env python3
"""
Clinical Reasoning Engine for Pixelated Empathy

Applies DSM-5 criteria and evidence-based intervention selection
based on extracted psychology knowledge.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class ClinicalAssessment:
    """Clinical assessment result."""
    presentation_id: str
    symptoms: List[str]
    potential_diagnoses: List[Dict[str, Any]]
    recommended_interventions: List[Dict[str, Any]]
    crisis_risk_level: str
    confidence_score: float
    reasoning: str
    contraindications: List[str]
    referral_recommendations: List[str]

@dataclass
class InterventionRecommendation:
    """Therapeutic intervention recommendation."""
    intervention_name: str
    modality: str
    rationale: str
    evidence_strength: str
    expert_support: List[str]
    application_steps: List[str]
    contraindications: List[str]
    expected_outcomes: List[str]

class ClinicalReasoningEngine:
    """Engine for clinical reasoning and intervention recommendation."""
    
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base = self._load_knowledge_base()
        self.logger = logging.getLogger(__name__)
        
        # DSM-5 symptom clusters
        self.dsm5_criteria = self._load_dsm5_criteria()
        self.intervention_protocols = self._load_intervention_protocols()
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load psychology knowledge base."""
        try:
            with open(self.knowledge_base_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
            return {}
    
    def _load_dsm5_criteria(self) -> Dict[str, Dict[str, Any]]:
        """Load DSM-5 diagnostic criteria."""
        return {
            'major_depressive_disorder': {
                'core_symptoms': [
                    'depressed mood',
                    'loss of interest or pleasure',
                    'significant weight loss or gain',
                    'insomnia or hypersomnia',
                    'psychomotor agitation or retardation',
                    'fatigue or loss of energy',
                    'feelings of worthlessness or guilt',
                    'difficulty concentrating',
                    'recurrent thoughts of death'
                ],
                'duration_criteria': '2 weeks minimum',
                'functional_impairment': True,
                'exclusions': ['manic episodes', 'mixed episodes']
            },
            'generalized_anxiety_disorder': {
                'core_symptoms': [
                    'excessive anxiety and worry',
                    'difficulty controlling worry',
                    'restlessness',
                    'easily fatigued',
                    'difficulty concentrating',
                    'irritability',
                    'muscle tension',
                    'sleep disturbance'
                ],
                'duration_criteria': '6 months minimum',
                'functional_impairment': True,
                'exclusions': ['substance use', 'medical condition']
            },
            'ptsd': {
                'core_symptoms': [
                    'intrusive memories',
                    'flashbacks',
                    'distressing dreams',
                    'avoidance of trauma reminders',
                    'negative alterations in cognition and mood',
                    'alterations in arousal and reactivity'
                ],
                'duration_criteria': '1 month minimum',
                'trauma_exposure': True,
                'functional_impairment': True
            },
            'borderline_personality_disorder': {
                'core_symptoms': [
                    'frantic efforts to avoid abandonment',
                    'unstable interpersonal relationships',
                    'identity disturbance',
                    'impulsivity',
                    'recurrent suicidal behavior',
                    'affective instability',
                    'chronic feelings of emptiness',
                    'inappropriate anger',
                    'stress-related paranoid ideation'
                ],
                'pattern_criteria': 'pervasive pattern since early adulthood',
                'functional_impairment': True
            }
        }
    
    def _load_intervention_protocols(self) -> Dict[str, Dict[str, Any]]:
        """Load evidence-based intervention protocols."""
        return {
            'cbt_depression': {
                'target_conditions': ['major_depressive_disorder', 'anxiety'],
                'evidence_level': 'strong',
                'techniques': [
                    'cognitive restructuring',
                    'behavioral activation',
                    'thought challenging',
                    'activity scheduling'
                ],
                'contraindications': ['active psychosis', 'severe cognitive impairment'],
                'expected_duration': '12-20 sessions'
            },
            'dbt_emotion_regulation': {
                'target_conditions': ['borderline_personality_disorder', 'emotional_dysregulation'],
                'evidence_level': 'strong',
                'techniques': [
                    'distress tolerance skills',
                    'emotion regulation skills',
                    'interpersonal effectiveness',
                    'mindfulness skills'
                ],
                'contraindications': ['unwillingness to commit to treatment'],
                'expected_duration': '1 year minimum'
            },
            'emdr_trauma': {
                'target_conditions': ['ptsd', 'complex_trauma'],
                'evidence_level': 'strong',
                'techniques': [
                    'bilateral stimulation',
                    'resource installation',
                    'reprocessing traumatic memories'
                ],
                'contraindications': ['unstable psychiatric condition', 'severe dissociation'],
                'expected_duration': 'variable, 6-12 sessions typical'
            },
            'grounding_techniques': {
                'target_conditions': ['anxiety', 'ptsd', 'panic_disorder'],
                'evidence_level': 'moderate',
                'techniques': [
                    '5-4-3-2-1 sensory grounding',
                    'progressive muscle relaxation',
                    'deep breathing exercises'
                ],
                'contraindications': [],
                'expected_duration': 'immediate relief technique'
            }
        }
    
    def analyze_presentation(self, symptoms: List[str], context: Dict[str, Any]) -> ClinicalAssessment:
        """Analyze clinical presentation and suggest interventions."""
        presentation_id = f"assessment_{hash(str(symptoms))}"
        
        # Assess potential diagnoses
        potential_diagnoses = self._assess_diagnoses(symptoms, context)
        
        # Determine crisis risk
        crisis_risk = self._assess_crisis_risk(symptoms, context)
        
        # Recommend interventions
        interventions = self._recommend_interventions(potential_diagnoses, symptoms, context)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(symptoms, potential_diagnoses, interventions)
        
        # Check contraindications
        contraindications = self._check_contraindications(interventions, context)
        
        # Generate referral recommendations
        referrals = self._generate_referrals(crisis_risk, potential_diagnoses, context)
        
        # Calculate confidence
        confidence = self._calculate_confidence(symptoms, potential_diagnoses)
        
        return ClinicalAssessment(
            presentation_id=presentation_id,
            symptoms=symptoms,
            potential_diagnoses=potential_diagnoses,
            recommended_interventions=interventions,
            crisis_risk_level=crisis_risk,
            confidence_score=confidence,
            reasoning=reasoning,
            contraindications=contraindications,
            referral_recommendations=referrals
        )
    
    def _assess_diagnoses(self, symptoms: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Assess potential DSM-5 diagnoses based on symptoms."""
        diagnoses = []
        symptoms_lower = [s.lower() for s in symptoms]
        
        for diagnosis, criteria in self.dsm5_criteria.items():
            match_score = self._calculate_diagnostic_match(symptoms_lower, criteria)
            
            if match_score > 0.3:  # Threshold for consideration
                diagnoses.append({
                    'diagnosis': diagnosis,
                    'match_score': match_score,
                    'matched_symptoms': self._get_matched_symptoms(symptoms_lower, criteria),
                    'missing_criteria': self._get_missing_criteria(symptoms_lower, criteria),
                    'confidence': self._calculate_diagnostic_confidence(match_score, criteria)
                })
        
        # Sort by match score
        return sorted(diagnoses, key=lambda x: x['match_score'], reverse=True)
    
    def _calculate_diagnostic_match(self, symptoms: List[str], criteria: Dict[str, Any]) -> float:
        """Calculate how well symptoms match diagnostic criteria."""
        core_symptoms = criteria.get('core_symptoms', [])
        if not core_symptoms:
            return 0.0
        
        matches = 0
        for symptom in symptoms:
            for core_symptom in core_symptoms:
                if any(word in symptom for word in core_symptom.lower().split()):
                    matches += 1
                    break
        
        return matches / len(core_symptoms)
    
    def _get_matched_symptoms(self, symptoms: List[str], criteria: Dict[str, Any]) -> List[str]:
        """Get symptoms that match diagnostic criteria."""
        matched = []
        core_symptoms = criteria.get('core_symptoms', [])
        
        for symptom in symptoms:
            for core_symptom in core_symptoms:
                if any(word in symptom for word in core_symptom.lower().split()):
                    matched.append(symptom)
                    break
        
        return matched
    
    def _get_missing_criteria(self, symptoms: List[str], criteria: Dict[str, Any]) -> List[str]:
        """Get diagnostic criteria not met by current symptoms."""
        core_symptoms = criteria.get('core_symptoms', [])
        missing = []
        
        for core_symptom in core_symptoms:
            found = False
            for symptom in symptoms:
                if any(word in symptom for word in core_symptom.lower().split()):
                    found = True
                    break
            if not found:
                missing.append(core_symptom)
        
        return missing
    
    def _assess_crisis_risk(self, symptoms: List[str], context: Dict[str, Any]) -> str:
        """Assess crisis risk level."""
        crisis_indicators = [
            'suicidal', 'suicide', 'self-harm', 'cutting', 'overdose',
            'harm others', 'violence', 'psychosis', 'hallucinations'
        ]
        
        symptoms_text = ' '.join(symptoms).lower()
        
        high_risk_count = sum(1 for indicator in crisis_indicators if indicator in symptoms_text)
        
        if high_risk_count >= 2:
            return 'HIGH'
        elif high_risk_count == 1:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _recommend_interventions(self, diagnoses: List[Dict[str, Any]], 
                               symptoms: List[str], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend evidence-based interventions."""
        recommendations = []
        
        # Get top diagnoses
        top_diagnoses = [d['diagnosis'] for d in diagnoses[:3]]
        
        for protocol_name, protocol in self.intervention_protocols.items():
            # Check if intervention targets any of the diagnosed conditions
            if any(condition in top_diagnoses for condition in protocol['target_conditions']):
                
                # Get expert support from knowledge base
                expert_support = self._get_expert_support(protocol_name)
                
                recommendations.append({
                    'intervention': protocol_name,
                    'evidence_level': protocol['evidence_level'],
                    'techniques': protocol['techniques'],
                    'target_conditions': protocol['target_conditions'],
                    'expert_support': expert_support,
                    'expected_duration': protocol['expected_duration'],
                    'contraindications': protocol['contraindications']
                })
        
        return recommendations
    
    def _get_expert_support(self, intervention_name: str) -> List[str]:
        """Get expert quotes supporting this intervention."""
        support = []
        
        # Search knowledge base for expert quotes about this intervention
        techniques = self.knowledge_base.get('techniques', {})
        
        for technique_id, technique in techniques.items():
            if intervention_name.lower() in technique.get('name', '').lower():
                expert_quotes = technique.get('expert_quotes', [])
                support.extend(expert_quotes[:2])  # Limit to 2 quotes
        
        return support
    
    def _generate_reasoning(self, symptoms: List[str], diagnoses: List[Dict[str, Any]], 
                          interventions: List[Dict[str, Any]]) -> str:
        """Generate clinical reasoning explanation."""
        reasoning_parts = []
        
        reasoning_parts.append(f"Based on {len(symptoms)} presenting symptoms:")
        reasoning_parts.append(f"- {', '.join(symptoms[:5])}")  # Show first 5 symptoms
        
        if diagnoses:
            top_diagnosis = diagnoses[0]
            reasoning_parts.append(f"Primary consideration: {top_diagnosis['diagnosis']} (match: {top_diagnosis['match_score']:.2f})")
        
        if interventions:
            reasoning_parts.append(f"Recommended approaches: {', '.join([i['intervention'] for i in interventions[:3]])}")
        
        return '\n'.join(reasoning_parts)
    
    def _check_contraindications(self, interventions: List[Dict[str, Any]], 
                               context: Dict[str, Any]) -> List[str]:
        """Check for contraindications to recommended interventions."""
        contraindications = []
        
        for intervention in interventions:
            contras = intervention.get('contraindications', [])
            contraindications.extend(contras)
        
        return list(set(contraindications))  # Remove duplicates
    
    def _generate_referrals(self, crisis_risk: str, diagnoses: List[Dict[str, Any]], 
                          context: Dict[str, Any]) -> List[str]:
        """Generate referral recommendations."""
        referrals = []
        
        if crisis_risk == 'HIGH':
            referrals.append('Immediate psychiatric evaluation')
            referrals.append('Crisis intervention team')
            referrals.append('Emergency department if imminent danger')
        
        elif crisis_risk == 'MODERATE':
            referrals.append('Psychiatric consultation within 48 hours')
            referrals.append('Safety planning with mental health professional')
        
        # Add specialty referrals based on diagnoses
        for diagnosis in diagnoses[:2]:  # Top 2 diagnoses
            diagnosis_name = diagnosis['diagnosis']
            
            if 'trauma' in diagnosis_name or 'ptsd' in diagnosis_name:
                referrals.append('Trauma specialist')
            
            if 'personality' in diagnosis_name:
                referrals.append('DBT-trained therapist')
            
            if 'substance' in diagnosis_name:
                referrals.append('Addiction specialist')
        
        return list(set(referrals))  # Remove duplicates
    
    def _calculate_diagnostic_confidence(self, match_score: float, criteria: Dict[str, Any]) -> float:
        """Calculate confidence in diagnostic assessment."""
        base_confidence = match_score
        
        # Adjust based on number of criteria met
        if match_score > 0.7:
            base_confidence += 0.1
        elif match_score < 0.4:
            base_confidence -= 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def _calculate_confidence(self, symptoms: List[str], diagnoses: List[Dict[str, Any]]) -> float:
        """Calculate overall assessment confidence."""
        if not diagnoses:
            return 0.3
        
        top_diagnosis_confidence = diagnoses[0].get('confidence', 0.5)
        symptom_specificity = min(1.0, len(symptoms) / 10.0)  # More symptoms = higher confidence up to 10
        
        return (top_diagnosis_confidence + symptom_specificity) / 2

    def generate_treatment_plan(self, assessment: ClinicalAssessment) -> Dict[str, Any]:
        """Generate comprehensive treatment plan based on assessment."""
        plan = {
            'assessment_summary': assessment.reasoning,
            'primary_targets': [],
            'intervention_sequence': [],
            'timeline': {},
            'monitoring_plan': {},
            'crisis_protocol': {}
        }
        
        # Primary treatment targets
        if assessment.potential_diagnoses:
            plan['primary_targets'] = [d['diagnosis'] for d in assessment.potential_diagnoses[:2]]
        
        # Intervention sequence based on evidence and safety
        interventions = sorted(assessment.recommended_interventions, 
                             key=lambda x: {'strong': 3, 'moderate': 2, 'weak': 1}.get(x['evidence_level'], 0),
                             reverse=True)
        
        plan['intervention_sequence'] = [
            {
                'phase': i + 1,
                'intervention': intervention['intervention'],
                'duration': intervention['expected_duration'],
                'goals': self._generate_treatment_goals(intervention, assessment.symptoms)
            }
            for i, intervention in enumerate(interventions[:3])
        ]
        
        # Crisis protocol
        if assessment.crisis_risk_level != 'LOW':
            plan['crisis_protocol'] = {
                'risk_level': assessment.crisis_risk_level,
                'safety_planning': True,
                'emergency_contacts': True,
                'monitoring_frequency': 'weekly' if assessment.crisis_risk_level == 'MODERATE' else 'daily'
            }
        
        return plan
    
    def _generate_treatment_goals(self, intervention: Dict[str, Any], symptoms: List[str]) -> List[str]:
        """Generate specific treatment goals for intervention."""
        goals = []
        
        intervention_name = intervention['intervention']
        
        if 'cbt' in intervention_name:
            goals.extend([
                'Identify and challenge negative thought patterns',
                'Increase behavioral activation and pleasant activities',
                'Develop coping strategies for mood symptoms'
            ])
        
        elif 'dbt' in intervention_name:
            goals.extend([
                'Learn distress tolerance skills',
                'Improve emotion regulation',
                'Enhance interpersonal effectiveness'
            ])
        
        elif 'emdr' in intervention_name or 'trauma' in intervention_name:
            goals.extend([
                'Process traumatic memories safely',
                'Reduce trauma-related symptoms',
                'Develop internal resources and coping'
            ])
        
        return goals


if __name__ == "__main__":
    # Example usage
    engine = ClinicalReasoningEngine("ai/pixel/knowledge/enhanced_psychology_knowledge_base.json")
    
    # Example clinical presentation
    symptoms = [
        "persistent sad mood",
        "loss of interest in activities",
        "difficulty sleeping",
        "fatigue and low energy",
        "difficulty concentrating",
        "feelings of worthlessness"
    ]
    
    context = {
        "duration": "3 weeks",
        "functional_impairment": True,
        "previous_episodes": False
    }
    
    assessment = engine.analyze_presentation(symptoms, context)
    print(f"Assessment: {assessment.presentation_id}")
    print(f"Crisis Risk: {assessment.crisis_risk_level}")
    print(f"Confidence: {assessment.confidence_score:.2f}")
    print(f"Reasoning: {assessment.reasoning}")
    
    if assessment.potential_diagnoses:
        print(f"Top diagnosis: {assessment.potential_diagnoses[0]['diagnosis']}")
    
    treatment_plan = engine.generate_treatment_plan(assessment)
    print(f"Treatment plan generated with {len(treatment_plan['intervention_sequence'])} phases")