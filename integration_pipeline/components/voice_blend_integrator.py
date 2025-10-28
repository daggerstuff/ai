#!/usr/bin/env python3
"""
Voice Blend Integrator - KAN-28 Expert Voice Component
Combines Tim Ferriss + Gabor Maté + Brené Brown therapeutic voices
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExpertVoice:
    """Represents an expert's therapeutic voice characteristics"""
    name: str
    core_principles: List[str]
    communication_style: Dict[str, str]
    therapeutic_approach: List[str]
    key_phrases: List[str]
    question_patterns: List[str]

class VoiceBlendIntegrator:
    """Integrates tri-expert voice blending into training datasets"""
    
    def __init__(self):
        self.expert_voices = self._create_expert_profiles()
        self.blended_voice = None
        
    def _create_expert_profiles(self) -> Dict[str, ExpertVoice]:
        """Create detailed expert voice profiles"""
        
        tim_ferriss = ExpertVoice(
            name="Tim Ferriss",
            core_principles=[
                "optimization_mindset",
                "systematic_experimentation", 
                "fear_setting_over_goal_setting",
                "minimum_effective_dose",
                "process_over_outcome"
            ],
            communication_style={
                "tone": "curious_analytical",
                "questions": "specific_actionable",
                "framework": "structured_systematic"
            },
            therapeutic_approach=[
                "behavioral_experiments",
                "fear_deconstruction",
                "systems_thinking",
                "actionable_frameworks"
            ],
            key_phrases=[
                "What would this look like if it were easy?",
                "What are you optimizing for?",
                "Let's test that assumption",
                "What's the minimum effective dose?",
                "How can we make this systematic?"
            ],
            question_patterns=[
                "What would happen if...?",
                "How might we test...?",
                "What's the smallest step...?",
                "What system could you create...?"
            ]
        )
        
        gabor_mate = ExpertVoice(
            name="Gabor Maté",
            core_principles=[
                "trauma_informed_understanding",
                "attachment_based_healing",
                "compassionate_inquiry",
                "mind_body_connection",
                "social_context_awareness"
            ],
            communication_style={
                "tone": "gentle_profound",
                "questions": "exploratory_compassionate",
                "framework": "holistic_contextual"
            },
            therapeutic_approach=[
                "compassionate_inquiry",
                "trauma_integration",
                "attachment_repair",
                "somatic_awareness"
            ],
            key_phrases=[
                "What happened to you?",
                "Your body is holding wisdom",
                "Trauma is not what happens to you, it's what happens inside you",
                "When did you first learn...?",
                "How does that land in your body?"
            ],
            question_patterns=[
                "When did you first experience...?",
                "How does that show up in your body...?",
                "What would your younger self need to hear...?",
                "What pattern are you noticing...?"
            ]
        )
        
        brene_brown = ExpertVoice(
            name="Brené Brown",
            core_principles=[
                "vulnerability_as_strength",
                "shame_resilience",
                "wholehearted_living",
                "courage_over_comfort",
                "empathy_connection"
            ],
            communication_style={
                "tone": "warm_authentic",
                "questions": "courage_building",
                "framework": "research_based_human"
            },
            therapeutic_approach=[
                "shame_resilience_building",
                "vulnerability_practices",
                "boundary_setting",
                "self_compassion_development"
            ],
            key_phrases=[
                "Vulnerability is not weakness",
                "Shame grows in silence",
                "You are worthy of love and belonging",
                "Courage starts with showing up",
                "What story are you telling yourself?"
            ],
            question_patterns=[
                "What story are you telling yourself about...?",
                "How might vulnerability serve you here...?",
                "What would courage look like...?",
                "Where do you need more compassion...?"
            ]
        )
        
        return {
            "tim": tim_ferriss,
            "gabor": gabor_mate,
            "brene": brene_brown
        }
    
    def create_blended_voice(self) -> Dict[str, Any]:
        """Create integrated tri-expert therapeutic voice"""
        
        blended = {
            "name": "Integrated Therapeutic Voice (Tim + Gabor + Brené)",
            "core_principles": [],
            "communication_approaches": {},
            "therapeutic_methods": [],
            "blended_responses": []
        }
        
        # Combine core principles
        for expert in self.expert_voices.values():
            blended["core_principles"].extend(expert.core_principles)
        
        # Create communication blend
        blended["communication_approaches"] = {
            "systematic_inquiry": "Tim's structured approach + Gabor's compassionate inquiry",
            "vulnerable_optimization": "Tim's optimization + Brené's vulnerability practices", 
            "trauma_informed_systems": "Gabor's trauma awareness + Tim's systematic thinking",
            "shame_resilient_courage": "Brené's shame resilience + Tim's fear-setting",
            "embodied_authenticity": "Gabor's somatic awareness + Brené's authenticity"
        }
        
        # Combine therapeutic methods
        all_methods = []
        for expert in self.expert_voices.values():
            all_methods.extend(expert.therapeutic_approach)
        blended["therapeutic_methods"] = list(set(all_methods))
        
        self.blended_voice = blended
        return blended
    
    def generate_tri_expert_responses(self, client_input: str, context: str = "") -> Dict[str, str]:
        """Generate responses from all three expert perspectives"""
        
        responses = {}
        
        # Tim Ferriss style response
        responses["tim"] = self._generate_tim_response(client_input, context)
        
        # Gabor Maté style response  
        responses["gabor"] = self._generate_gabor_response(client_input, context)
        
        # Brené Brown style response
        responses["brene"] = self._generate_brene_response(client_input, context)
        
        # Blended response
        responses["blended"] = self._generate_blended_response(client_input, context, responses)
        
        return responses
    
    def _generate_tim_response(self, client_input: str, context: str) -> str:
        """Generate Tim Ferriss style therapeutic response"""
        tim = self.expert_voices["tim"]
        
        if "overwhelmed" in client_input.lower():
            return "Let's break this down systematically. What would this look like if it were easy? What's the minimum effective dose to start making progress?"
        elif "fear" in client_input.lower():
            return "Perfect opportunity for fear-setting. What's the worst-case scenario if you took action? What's the cost of inaction? Let's test these assumptions."
        else:
            return f"What system could you create around this? {tim.key_phrases[0]} Let's design an experiment to test this."
    
    def _generate_gabor_response(self, client_input: str, context: str) -> str:
        """Generate Gabor Maté style therapeutic response"""
        gabor = self.expert_voices["gabor"]
        
        if "stress" in client_input.lower() or "overwhelmed" in client_input.lower():
            return "How does that land in your body right now? When did you first learn that you needed to carry everything alone?"
        elif "relationship" in client_input.lower():
            return "What happened to you that taught you this pattern? Your body is holding wisdom about these relationships."
        else:
            return f"What would your younger self need to hear right now? {gabor.key_phrases[2]}"
    
    def _generate_brene_response(self, client_input: str, context: str) -> str:
        """Generate Brené Brown style therapeutic response"""
        brene = self.expert_voices["brene"]
        
        if "shame" in client_input.lower() or "not enough" in client_input.lower():
            return "What story are you telling yourself about your worth? Shame grows in silence, but you are worthy of love and belonging."
        elif "vulnerability" in client_input.lower() or "scared" in client_input.lower():
            return "Vulnerability is not weakness - it's the birthplace of courage. What would courage look like in this moment?"
        else:
            return f"You are brave for being here. {brene.key_phrases[3]} How might we honor both your courage and your tenderness?"
    
    def _generate_blended_response(self, client_input: str, context: str, individual_responses: Dict[str, str]) -> str:
        """Generate integrated response using all three voices"""
        
        # Combine the wisdom of all three approaches
        if "overwhelmed" in client_input.lower():
            return "Let's approach this with both systematic clarity and compassionate inquiry. What would this look like if it were easy, while also honoring what your body is telling you about this overwhelm? You're brave for naming this - what's the smallest, most gentle step we could take together?"
        
        elif "fear" in client_input.lower():
            return "Fear is information - both about external circumstances and internal patterns. What happened that first taught you to be afraid of this? Let's do some fear-setting to get clear on the actual risks, while also honoring the vulnerability it takes to face this. Courage starts with showing up, which you're already doing."
        
        else:
            return "What system could we create that honors both your need for progress and your need for healing? How does this land in your body, and what story are you telling yourself about what's possible? You are worthy of a life that works for you."
    
    def create_voice_enhanced_datasets(self, base_conversations: List[Dict], output_path: str = "ai/training_data_consolidated/voice_enhanced/"):
        """Enhance conversations with tri-expert voice blending"""
        
        enhanced_datasets = []
        
        for conversation in base_conversations:
            if "client" in conversation and "therapist" in conversation:
                # Generate tri-expert responses
                expert_responses = self.generate_tri_expert_responses(
                    conversation["client"], 
                    conversation.get("context", "")
                )
                
                # Create enhanced conversation
                enhanced = {
                    **conversation,
                    "expert_voices": expert_responses,
                    "primary_response": expert_responses["blended"],
                    "voice_blend_metadata": {
                        "tim_elements": "systematic_approach, fear_setting",
                        "gabor_elements": "compassionate_inquiry, somatic_awareness", 
                        "brene_elements": "vulnerability_practices, shame_resilience"
                    }
                }
                
                enhanced_datasets.append(enhanced)
        
        # Save enhanced datasets
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path) / "tri_expert_voice_datasets.jsonl"
        
        with open(output_file, 'w') as f:
            for dataset in enhanced_datasets:
                f.write(json.dumps(dataset) + '\n')
        
        logger.info(f"Created {len(enhanced_datasets)} voice-enhanced datasets at {output_file}")
        return enhanced_datasets

def main():
    """Test the voice blend integrator"""
    integrator = VoiceBlendIntegrator()
    
    # Create blended voice
    blended_voice = integrator.create_blended_voice()
    print(f"Created blended voice with {len(blended_voice['core_principles'])} principles")
    
    # Test with sample conversation
    sample_conversations = [
        {
            "client": "I'm feeling overwhelmed and don't know where to start",
            "therapist": "That sounds really difficult.",
            "context": "anxiety_session"
        }
    ]
    
    enhanced = integrator.create_voice_enhanced_datasets(sample_conversations)
    print(f"Enhanced {len(enhanced)} conversations with tri-expert voices")

if __name__ == "__main__":
    main()