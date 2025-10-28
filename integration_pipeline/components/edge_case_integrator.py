#!/usr/bin/env python3
"""
Edge Case Integrator - KAN-28 Component #3
Integrates nightmare fuel scenarios with expert voices and therapeutic frameworks
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EdgeCaseScenario:
    """Represents a challenging therapeutic edge case"""
    scenario_type: str
    severity_level: int  # 1-10 scale
    client_presentation: str
    expert_response_needed: List[str]
    safety_considerations: List[str]
    therapeutic_goals: List[str]

class EdgeCaseIntegrator:
    """Integrates edge case scenarios with expert therapeutic responses"""
    
    def __init__(self, edge_case_dir: str = "ai/training_data_consolidated/edge_cases/"):
        self.edge_case_dir = Path(edge_case_dir)
        self.scenarios = []
        
    def load_existing_edge_cases(self) -> List[Dict[str, Any]]:
        """Load existing edge case scenarios from directory"""
        
        edge_cases = []
        
        # Check if directory exists
        if not self.edge_case_dir.exists():
            logger.warning(f"Edge case directory not found: {self.edge_case_dir}")
            # Create sample edge cases
            return self._create_sample_edge_cases()
        
        # Load any existing edge case files
        for file_path in self.edge_case_dir.glob("*.json*"):
            try:
                with open(file_path, 'r') as f:
                    if file_path.suffix == '.jsonl':
                        for line in f:
                            edge_cases.append(json.loads(line.strip()))
                    else:
                        edge_cases.extend(json.load(f))
            except Exception as e:
                logger.warning(f"Could not load edge case file {file_path}: {e}")
        
        if not edge_cases:
            edge_cases = self._create_sample_edge_cases()
            
        return edge_cases
    
    def _create_sample_edge_cases(self) -> List[Dict[str, Any]]:
        """Create sample edge case scenarios for testing"""
        
        scenarios = [
            {
                "scenario_type": "suicidal_ideation",
                "severity_level": 9,
                "client_presentation": "I've been thinking about ending my life. I have a plan and I don't see any other way out.",
                "safety_considerations": ["immediate_risk_assessment", "safety_planning", "emergency_protocols"],
                "therapeutic_goals": ["establish_safety", "build_hope", "create_support_network"]
            },
            {
                "scenario_type": "trauma_flashback",
                "severity_level": 8,
                "client_presentation": "I'm having flashbacks again. I can't tell what's real anymore. I feel like I'm back there.",
                "safety_considerations": ["grounding_techniques", "present_moment_awareness", "safety_validation"],
                "therapeutic_goals": ["establish_present_safety", "trauma_integration", "nervous_system_regulation"]
            },
            {
                "scenario_type": "severe_dissociation",
                "severity_level": 7,
                "client_presentation": "I feel like I'm watching myself from outside my body. Nothing feels real. I can't connect to anything.",
                "safety_considerations": ["grounding_interventions", "co_regulation", "gentle_presence"],
                "therapeutic_goals": ["embodied_presence", "nervous_system_safety", "gradual_integration"]
            },
            {
                "scenario_type": "relationship_crisis",
                "severity_level": 6,
                "client_presentation": "My partner left me and I feel completely worthless. I don't know who I am without them.",
                "safety_considerations": ["attachment_wounds", "identity_stability", "support_systems"],
                "therapeutic_goals": ["identity_rebuilding", "attachment_healing", "self_worth_restoration"]
            },
            {
                "scenario_type": "addiction_relapse",
                "severity_level": 8,
                "client_presentation": "I relapsed last night after 6 months sober. I feel like a complete failure and want to give up.",
                "safety_considerations": ["relapse_prevention", "shame_spirals", "motivation_rebuilding"],
                "therapeutic_goals": ["shame_resilience", "progress_reframing", "recovery_recommitment"]
            }
        ]
        
        return scenarios
    
    def integrate_with_expert_voices(self, edge_cases: List[Dict], expert_voices: Dict) -> List[Dict[str, Any]]:
        """Integrate edge cases with tri-expert therapeutic responses"""
        
        integrated_scenarios = []
        
        for case in edge_cases:
            # Generate expert responses for this edge case
            expert_responses = self._generate_expert_responses_for_case(case, expert_voices)
            
            # Create integrated scenario
            integrated = {
                **case,
                "expert_responses": expert_responses,
                "integrated_response": self._create_integrated_response(case, expert_responses),
                "safety_protocol": self._create_safety_protocol(case),
                "therapeutic_framework": self._create_therapeutic_framework(case)
            }
            
            integrated_scenarios.append(integrated)
        
        return integrated_scenarios
    
    def _generate_expert_responses_for_case(self, case: Dict, expert_voices: Dict) -> Dict[str, str]:
        """Generate responses from each expert for the edge case"""
        
        client_presentation = case.get("client_presentation", "")
        scenario_type = case.get("scenario_type", "")
        
        responses = {}
        
        # Tim Ferriss approach - systematic, fear-setting, actionable
        if scenario_type == "suicidal_ideation":
            responses["tim"] = "Right now, we need to create a systematic safety plan. What would need to happen in the next 24 hours for you to feel 1% safer? Let's design the minimum effective dose of support."
        elif scenario_type == "trauma_flashback":
            responses["tim"] = "Your brain is trying to protect you with outdated information. What would grounding look like if it were easy? Let's create a simple system you can use."
        else:
            responses["tim"] = "What's the smallest step we could take right now that would move you 1% in the direction of safety? Let's make this systematic and achievable."
        
        # Gabor Maté approach - trauma-informed, compassionate inquiry
        if scenario_type == "suicidal_ideation":
            responses["gabor"] = "Your pain is real and valid. What happened to you that taught you that your life doesn't matter? Your body is trying to protect you - let's listen to what it needs."
        elif scenario_type == "trauma_flashback":
            responses["gabor"] = "Your nervous system is responding to old wounds. When did you first learn that the world wasn't safe? Let's help your body remember that you're here now, not back there."
        else:
            responses["gabor"] = "What happened to you? Your pain makes sense in the context of your story. Let's explore this with compassion for the part of you that's struggling."
        
        # Brené Brown approach - shame resilience, vulnerability, courage
        if scenario_type == "suicidal_ideation":
            responses["brene"] = "You are worthy of love and belonging, even in this dark moment. Shame grows in silence - thank you for having the courage to share this with me."
        elif scenario_type == "trauma_flashback":
            responses["brene"] = "You're being incredibly brave by staying present with me right now. Vulnerability is not weakness - it's your pathway back to wholeness."
        else:
            responses["brene"] = "What story are you telling yourself about your worth right now? You belong here, and your struggle doesn't define your value."
        
        return responses
    
    def _create_integrated_response(self, case: Dict, expert_responses: Dict) -> str:
        """Create a blended response using all three expert approaches"""
        
        scenario_type = case.get("scenario_type", "")
        
        if scenario_type == "suicidal_ideation":
            return "Your pain is real and you matter. Let's create a safety plan together - what would the next 24 hours look like if we designed them for your wellbeing? You're being incredibly brave by sharing this. Your life has value, and we're going to take this one small step at a time."
        
        elif scenario_type == "trauma_flashback":
            return "Your nervous system is doing its job - trying to protect you. Let's help your body remember you're safe here with me now. What would grounding feel like if it were gentle and easy? You're showing incredible courage by staying present."
        
        else:
            return "What you're experiencing makes complete sense given your story. Let's approach this with both compassion and practical steps. You are worthy of support, and we're going to figure this out together, one small step at a time."
    
    def _create_safety_protocol(self, case: Dict) -> Dict[str, Any]:
        """Create safety protocol for the edge case"""
        
        return {
            "immediate_actions": case.get("safety_considerations", []),
            "risk_level": case.get("severity_level", 5),
            "emergency_contacts": ["crisis_hotline", "emergency_services", "trusted_support"],
            "follow_up_required": case.get("severity_level", 5) >= 7
        }
    
    def _create_therapeutic_framework(self, case: Dict) -> Dict[str, Any]:
        """Create therapeutic framework for the edge case"""
        
        return {
            "primary_goals": case.get("therapeutic_goals", []),
            "therapeutic_modalities": ["trauma_informed", "attachment_based", "somatic_experiencing"],
            "session_structure": "safety_first_then_processing",
            "progress_markers": ["safety_increase", "symptom_reduction", "functional_improvement"]
        }
    
    def create_edge_case_datasets(self, output_path: str = "ai/training_data_consolidated/edge_cases_enhanced/") -> List[Dict[str, Any]]:
        """Create integrated edge case datasets"""
        
        # Load existing edge cases
        edge_cases = self.load_existing_edge_cases()
        
        # For now, use simplified expert voices structure
        expert_voices = {
            "tim": {"style": "systematic_actionable"},
            "gabor": {"style": "trauma_informed_compassionate"},
            "brene": {"style": "shame_resilient_vulnerable"}
        }
        
        # Integrate with expert voices
        integrated_datasets = self.integrate_with_expert_voices(edge_cases, expert_voices)
        
        # Save datasets
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path) / "edge_cases_integrated.jsonl"
        
        with open(output_file, 'w') as f:
            for dataset in integrated_datasets:
                f.write(json.dumps(dataset) + '\n')
        
        logger.info(f"Created {len(integrated_datasets)} edge case integrated datasets at {output_file}")
        return integrated_datasets

def main():
    """Test the edge case integrator"""
    integrator = EdgeCaseIntegrator()
    datasets = integrator.create_edge_case_datasets()
    print(f"Generated {len(datasets)} edge case integrated datasets")

if __name__ == "__main__":
    main()