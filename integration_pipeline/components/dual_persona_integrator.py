#!/usr/bin/env python3
"""
Dual Persona Integrator - KAN-28 Component #4
Integrates therapist/client dynamics with realistic therapeutic relationships
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class TherapeuticPersona:
    """Represents a therapeutic persona (therapist or client)"""
    role: str
    characteristics: List[str]
    communication_style: str
    emotional_patterns: List[str]
    growth_trajectory: List[str]

class DualPersonaIntegrator:
    """Integrates dual persona training with therapeutic relationship dynamics"""
    
    def __init__(self, dual_persona_dir: str = "ai/pipelines/dual_persona_training/"):
        self.dual_persona_dir = Path(dual_persona_dir)
        self.client_personas = []
        self.therapist_personas = []
        
    def load_existing_personas(self) -> Dict[str, List[Dict]]:
        """Load existing dual persona data"""
        
        personas = {"clients": [], "therapists": []}
        
        # Check if directory exists
        if not self.dual_persona_dir.exists():
            logger.warning(f"Dual persona directory not found: {self.dual_persona_dir}")
            return self._create_sample_personas()
        
        # Load any existing persona files
        for file_path in self.dual_persona_dir.glob("*.json*"):
            try:
                with open(file_path, 'r') as f:
                    if file_path.suffix == '.jsonl':
                        for line in f:
                            data = json.loads(line.strip())
                            if 'client' in data:
                                personas["clients"].append(data)
                            if 'therapist' in data:
                                personas["therapists"].append(data)
                    else:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                if 'client' in item:
                                    personas["clients"].append(item)
                                if 'therapist' in item:
                                    personas["therapists"].append(item)
            except Exception as e:
                logger.warning(f"Could not load persona file {file_path}: {e}")
        
        if not personas["clients"] and not personas["therapists"]:
            personas = self._create_sample_personas()
            
        return personas
    
    def _create_sample_personas(self) -> Dict[str, List[Dict]]:
        """Create sample client and therapist personas"""
        
        client_personas = [
            {
                "client_id": "anxious_perfectionist",
                "demographics": {"age": 28, "background": "high_achiever"},
                "presentation": {
                    "primary_concerns": ["anxiety", "perfectionism", "burnout"],
                    "communication_style": "analytical_but_emotional",
                    "defense_mechanisms": ["intellectualization", "people_pleasing"],
                    "emotional_patterns": ["anxiety_spirals", "self_criticism", "overwhelm"]
                },
                "growth_trajectory": [
                    "building_self_awareness",
                    "learning_self_compassion", 
                    "setting_boundaries",
                    "accepting_imperfection"
                ],
                "therapeutic_needs": ["validation", "practical_tools", "emotional_regulation"]
            },
            {
                "client_id": "trauma_survivor",
                "demographics": {"age": 35, "background": "childhood_trauma"},
                "presentation": {
                    "primary_concerns": ["PTSD", "trust_issues", "dissociation"],
                    "communication_style": "guarded_initially_then_vulnerable",
                    "defense_mechanisms": ["hypervigilance", "emotional_numbing", "isolation"],
                    "emotional_patterns": ["triggered_responses", "shame_spirals", "disconnection"]
                },
                "growth_trajectory": [
                    "building_safety",
                    "developing_trust",
                    "processing_trauma",
                    "reclaiming_agency"
                ],
                "therapeutic_needs": ["safety", "paced_approach", "somatic_awareness"]
            },
            {
                "client_id": "relationship_struggles",
                "demographics": {"age": 42, "background": "attachment_wounds"},
                "presentation": {
                    "primary_concerns": ["relationship_patterns", "abandonment_fear", "intimacy_issues"],
                    "communication_style": "emotionally_expressive_but_reactive",
                    "defense_mechanisms": ["protest_behaviors", "withdrawal", "testing_boundaries"],
                    "emotional_patterns": ["attachment_anxiety", "fear_of_rejection", "anger_outbursts"]
                },
                "growth_trajectory": [
                    "understanding_patterns",
                    "developing_secure_attachment",
                    "improving_communication",
                    "building_healthy_relationships"
                ],
                "therapeutic_needs": ["consistent_presence", "attachment_repair", "relationship_skills"]
            }
        ]
        
        therapist_personas = [
            {
                "therapist_id": "integrative_approach",
                "theoretical_orientation": ["humanistic", "cognitive_behavioral", "trauma_informed"],
                "communication_style": {
                    "tone": "warm_professional",
                    "approach": "collaborative_curious",
                    "interventions": "balanced_support_challenge"
                },
                "strengths": ["empathic_attunement", "practical_tools", "trauma_sensitivity"],
                "growth_areas": ["working_with_resistance", "managing_countertransference"],
                "therapeutic_techniques": [
                    "reflective_listening",
                    "cognitive_restructuring", 
                    "somatic_interventions",
                    "attachment_repair"
                ]
            },
            {
                "therapist_id": "psychodynamic_focus",
                "theoretical_orientation": ["psychodynamic", "attachment_based", "depth_psychology"],
                "communication_style": {
                    "tone": "gentle_insightful",
                    "approach": "exploratory_interpretive",
                    "interventions": "insight_oriented"
                },
                "strengths": ["pattern_recognition", "unconscious_dynamics", "therapeutic_relationship"],
                "growth_areas": ["concrete_tools", "crisis_intervention"],
                "therapeutic_techniques": [
                    "interpretation",
                    "transference_work",
                    "dream_analysis",
                    "relational_dynamics"
                ]
            }
        ]
        
        return {"clients": client_personas, "therapists": therapist_personas}
    
    def create_therapeutic_relationships(self, personas: Dict) -> List[Dict[str, Any]]:
        """Create realistic therapeutic relationship dynamics"""
        
        relationships = []
        
        clients = personas.get("clients", [])
        therapists = personas.get("therapists", [])
        
        # Create client-therapist pairings
        for i, client in enumerate(clients):
            therapist = therapists[i % len(therapists)]  # Cycle through therapists
            
            relationship = self._develop_therapeutic_relationship(client, therapist)
            relationships.append(relationship)
        
        return relationships
    
    def _develop_therapeutic_relationship(self, client: Dict, therapist: Dict) -> Dict[str, Any]:
        """Develop a specific therapeutic relationship over time"""
        
        relationship = {
            "client_profile": client,
            "therapist_profile": therapist,
            "relationship_dynamics": self._create_relationship_dynamics(client, therapist),
            "session_progression": self._create_session_progression(client, therapist),
            "therapeutic_alliance": self._track_alliance_development(client, therapist)
        }
        
        return relationship
    
    def _create_relationship_dynamics(self, client: Dict, therapist: Dict) -> Dict[str, Any]:
        """Create the dynamics between specific client and therapist"""
        
        client_style = client.get("presentation", {}).get("communication_style", "")
        therapist_style = therapist.get("communication_style", {}).get("approach", "")
        
        # Determine compatibility and challenges
        if "analytical" in client_style and "collaborative" in therapist_style:
            compatibility = "high"
            dynamics = "intellectual_connection_with_emotional_growth"
        elif "guarded" in client_style and "gentle" in therapist_style:
            compatibility = "developing"
            dynamics = "slow_trust_building_with_breakthrough_moments"
        else:
            compatibility = "moderate"
            dynamics = "working_through_differences_to_find_connection"
        
        return {
            "compatibility_level": compatibility,
            "primary_dynamics": dynamics,
            "transference_patterns": self._identify_transference(client),
            "countertransference_patterns": self._identify_countertransference(therapist, client),
            "therapeutic_challenges": self._identify_challenges(client, therapist),
            "growth_opportunities": self._identify_opportunities(client, therapist)
        }
    
    def _create_session_progression(self, client: Dict, therapist: Dict) -> List[Dict[str, Any]]:
        """Create progression of therapy sessions"""
        
        sessions = []
        
        # Early sessions (1-5)
        for session_num in range(1, 6):
            sessions.append({
                "session_number": session_num,
                "phase": "assessment_rapport_building",
                "client_presentation": self._get_early_presentation(client, session_num),
                "therapist_response": self._get_early_response(therapist, client, session_num),
                "alliance_rating": 0.3 + (session_num * 0.1),
                "session_goals": ["build_rapport", "assess_needs", "establish_safety"]
            })
        
        # Middle sessions (6-15)
        for session_num in range(6, 16):
            sessions.append({
                "session_number": session_num,
                "phase": "working_through_issues",
                "client_presentation": self._get_middle_presentation(client, session_num),
                "therapist_response": self._get_middle_response(therapist, client, session_num),
                "alliance_rating": 0.6 + ((session_num - 5) * 0.03),
                "session_goals": ["process_trauma", "develop_skills", "challenge_patterns"]
            })
        
        # Later sessions (16-25)
        for session_num in range(16, 26):
            sessions.append({
                "session_number": session_num,
                "phase": "integration_termination_prep",
                "client_presentation": self._get_later_presentation(client, session_num),
                "therapist_response": self._get_later_response(therapist, client, session_num),
                "alliance_rating": 0.85 + ((session_num - 15) * 0.01),
                "session_goals": ["integrate_learning", "prepare_termination", "maintain_gains"]
            })
        
        return sessions
    
    def _get_early_presentation(self, client: Dict, session_num: int) -> str:
        """Get client presentation in early sessions"""
        client_id = client.get("client_id", "")
        
        if "anxious" in client_id:
            return f"Session {session_num}: I'm still not sure this will help. I've tried everything and I'm still anxious all the time."
        elif "trauma" in client_id:
            return f"Session {session_num}: I don't really know what to say. It's hard to trust that this is safe."
        else:
            return f"Session {session_num}: I keep having the same problems in my relationships. I don't know what's wrong with me."
    
    def _get_early_response(self, therapist: Dict, client: Dict, session_num: int) -> str:
        """Get therapist response in early sessions"""
        orientation = therapist.get("theoretical_orientation", [""])[0]
        
        if "humanistic" in orientation:
            return "I hear how much pain you're in. Let's take this slowly and figure out what you need to feel safe here."
        elif "psychodynamic" in orientation:
            return "These patterns you're describing - when do you remember first experiencing something similar?"
        else:
            return "Thank you for sharing that with me. What would it look like if therapy actually helped?"
    
    def _get_middle_presentation(self, client: Dict, session_num: int) -> str:
        """Get client presentation in middle sessions"""
        return f"Session {session_num}: I'm starting to see some patterns, but it's still really hard when I'm triggered."
    
    def _get_middle_response(self, therapist: Dict, client: Dict, session_num: int) -> str:
        """Get therapist response in middle sessions"""
        return "That awareness of the patterns is huge progress. Let's explore what happens right before you get triggered."
    
    def _get_later_presentation(self, client: Dict, session_num: int) -> str:
        """Get client presentation in later sessions"""
        return f"Session {session_num}: I used the tools we practiced and it actually helped. I'm starting to feel different."
    
    def _get_later_response(self, therapist: Dict, client: Dict, session_num: int) -> str:
        """Get therapist response in later sessions"""
        return "I can see the changes in you. How does it feel to know you have these tools now?"
    
    def _identify_transference(self, client: Dict) -> List[str]:
        """Identify likely transference patterns for client"""
        concerns = client.get("presentation", {}).get("primary_concerns", [])
        
        if "trust_issues" in str(concerns):
            return ["suspicious_of_motives", "testing_boundaries", "fear_of_abandonment"]
        elif "perfectionism" in str(concerns):
            return ["wanting_approval", "fear_of_disappointing", "intellectualizing_emotions"]
        else:
            return ["seeking_parent_figure", "fear_of_judgment", "ambivalent_attachment"]
    
    def _identify_countertransference(self, therapist: Dict, client: Dict) -> List[str]:
        """Identify likely countertransference patterns"""
        return ["protective_instincts", "wanting_to_fix", "frustrated_with_resistance"]
    
    def _identify_challenges(self, client: Dict, therapist: Dict) -> List[str]:
        """Identify therapeutic challenges"""
        return ["resistance_to_change", "trust_building", "managing_setbacks"]
    
    def _identify_opportunities(self, client: Dict, therapist: Dict) -> List[str]:
        """Identify growth opportunities"""
        return ["corrective_relationship_experience", "skill_development", "insight_building"]
    
    def _track_alliance_development(self, client: Dict, therapist: Dict) -> Dict[str, Any]:
        """Track therapeutic alliance development"""
        return {
            "initial_rating": 0.3,
            "peak_rating": 0.95,
            "alliance_factors": ["trust", "collaboration", "empathy", "goal_agreement"],
            "rupture_repair_cycles": ["early_testing", "middle_breakthrough", "late_integration"]
        }
    
    def create_dual_persona_datasets(self, output_path: str = "ai/training_data_consolidated/dual_persona_enhanced/") -> List[Dict[str, Any]]:
        """Create integrated dual persona datasets"""
        
        # Load personas
        personas = self.load_existing_personas()
        
        # Create therapeutic relationships
        relationships = self.create_therapeutic_relationships(personas)
        
        # Convert to training datasets
        datasets = []
        for relationship in relationships:
            sessions = relationship.get("session_progression", [])
            for session in sessions:
                dataset = {
                    "relationship_id": f"{relationship['client_profile']['client_id']}_{relationship['therapist_profile']['therapist_id']}",
                    "session_data": session,
                    "relationship_context": relationship["relationship_dynamics"],
                    "therapeutic_alliance": session.get("alliance_rating", 0.5),
                    "training_type": "dual_persona_integrated"
                }
                datasets.append(dataset)
        
        # Save datasets
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path) / "dual_persona_integrated.jsonl"
        
        with open(output_file, 'w') as f:
            for dataset in datasets:
                f.write(json.dumps(dataset) + '\n')
        
        logger.info(f"Created {len(datasets)} dual persona integrated datasets at {output_file}")
        return datasets

def main():
    """Test the dual persona integrator"""
    integrator = DualPersonaIntegrator()
    datasets = integrator.create_dual_persona_datasets()
    print(f"Generated {len(datasets)} dual persona integrated datasets")

if __name__ == "__main__":
    main()