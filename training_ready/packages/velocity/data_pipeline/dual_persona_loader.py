#!/usr/bin/env python3
"""
Dual Persona Training Data Loader
Loads multi-persona therapeutic interaction data
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger("dataset_pipeline.dual_persona_loader")


@dataclass
class DualPersonaDialogue:
    """Structured dual persona dialogue"""
    dialogue_id: str
    persona_1: str  # e.g., "empathetic_listener"
    persona_2: str  # e.g., "cognitive_restructurer"
    conversation: List[Dict]  # List of turns with persona labels
    scenario: str
    consistency_score: float = 0.0
    
    def to_training_format(self) -> Dict:
        """Convert to standard training format"""
        # Create conversational text from turns
        text_parts = []
        for turn in self.conversation:
            speaker = turn.get('speaker', 'Therapist')
            content = turn.get('content', '')
            text_parts.append(f"{speaker}: {content}")
        
        text = "\n".join(text_parts)
        
        # Extract prompt and response (last two turns)
        prompt = ""
        response = ""
        if len(self.conversation) >= 2:
            prompt = self.conversation[-2].get('content', '')
            response = self.conversation[-1].get('content', '')
        
        return {
            "text": text,
            "prompt": prompt,
            "response": response,
            "metadata": {
                "source": "dual_persona",
                "dialogue_id": self.dialogue_id,
                "persona_1": self.persona_1,
                "persona_2": self.persona_2,
                "scenario": self.scenario,
                "consistency_score": self.consistency_score,
                "is_multi_persona": True,
                "is_edge_case": False
            }
        }


class DualPersonaLoader:
    """Loader for dual persona training data"""
    
    def __init__(self, dual_persona_dir: str = "ai/pipelines/dual_persona_training"):
        self.pipeline_dir = Path(dual_persona_dir)
        self.training_file = self.pipeline_dir / "dual_persona_training_data.jsonl"
        self.config_file = self.pipeline_dir / "training_config.json"
        
    def load_dialogues(self) -> List[DualPersonaDialogue]:
        """Load dual persona dialogues"""
        if not self.training_file.exists():
            logger.warning(f"Dual persona training file not found: {self.training_file}")
            logger.info("Generating synthetic dual persona data...")
            return self._generate_synthetic_data()
        
        try:
            dialogues = []
            with open(self.training_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        dialogue = DualPersonaDialogue(
                            dialogue_id=data.get('dialogue_id', f"dual_{line_num:04d}"),
                            persona_1=data.get('persona_1', 'empathetic_listener'),
                            persona_2=data.get('persona_2', 'cognitive_restructurer'),
                            conversation=data.get('conversation', []),
                            scenario=data.get('scenario', ''),
                            consistency_score=data.get('consistency_score', 0.0)
                        )
                        dialogues.append(dialogue)
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Error parsing line {line_num}: {e}")
                        continue
            
            logger.info(f"Loaded {len(dialogues)} dual persona dialogues")
            return dialogues
            
        except Exception as e:
            logger.error(f"Failed to load dual persona data: {e}")
            return []
    
    def _generate_synthetic_data(self) -> List[DualPersonaDialogue]:
        """Generate synthetic dual persona training data"""
        logger.info("Generating synthetic dual persona dialogues...")
        
        personas = [
            ("empathetic_listener", "cognitive_restructurer"),
            ("crisis_interventionist", "supportive_counselor"),
            ("psychoeducator", "motivational_interviewer"),
            ("trauma_specialist", "mindfulness_guide")
        ]
        
        scenarios = [
            "Client expressing anxiety about work",
            "Client dealing with relationship conflict",
            "Client experiencing depression symptoms",
            "Client processing grief and loss",
            "Client managing stress and overwhelm"
        ]
        
        dialogues = []
        dialogue_id = 1
        
        for persona_pair in personas:
            for scenario in scenarios:
                # Generate a simple dialogue
                conversation = [
                    {
                        "speaker": "Therapist",
                        "persona": persona_pair[0],
                        "content": f"I understand you're going through a difficult time with {scenario.lower()}. Can you tell me more about what you're experiencing?"
                    },
                    {
                        "speaker": "Client",
                        "content": "I've been feeling really overwhelmed and don't know how to cope."
                    },
                    {
                        "speaker": "Therapist",
                        "persona": persona_pair[1],
                        "content": "Let's work together to identify some specific strategies that might help you manage these feelings. What has helped you in the past?"
                    },
                    {
                        "speaker": "Client",
                        "content": "Sometimes talking to friends helps, but lately I've been isolating myself."
                    }
                ]
                
                dialogue = DualPersonaDialogue(
                    dialogue_id=f"dual_synthetic_{dialogue_id:04d}",
                    persona_1=persona_pair[0],
                    persona_2=persona_pair[1],
                    conversation=conversation,
                    scenario=scenario,
                    consistency_score=0.85
                )
                dialogues.append(dialogue)
                dialogue_id += 1
        
        logger.info(f"Generated {len(dialogues)} synthetic dual persona dialogues")
        
        # Save for future use
        self._save_dialogues(dialogues)
        
        return dialogues
    
    def _save_dialogues(self, dialogues: List[DualPersonaDialogue]):
        """Save dialogues to file"""
        try:
            self.pipeline_dir.mkdir(parents=True, exist_ok=True)
            with open(self.training_file, 'w') as f:
                for dialogue in dialogues:
                    data = {
                        'dialogue_id': dialogue.dialogue_id,
                        'persona_1': dialogue.persona_1,
                        'persona_2': dialogue.persona_2,
                        'conversation': dialogue.conversation,
                        'scenario': dialogue.scenario,
                        'consistency_score': dialogue.consistency_score
                    }
                    f.write(json.dumps(data) + '\n')
            logger.info(f"Saved {len(dialogues)} dialogues to {self.training_file}")
        except Exception as e:
            logger.error(f"Failed to save dialogues: {e}")
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded dual persona data"""
        dialogues = self.load_dialogues()
        
        if not dialogues:
            return {
                "total_dialogues": 0,
                "persona_pairs": {},
                "scenarios": {},
                "avg_consistency_score": 0.0
            }
        
        # Count persona pairs
        persona_pairs = {}
        for dialogue in dialogues:
            pair = f"{dialogue.persona_1} + {dialogue.persona_2}"
            persona_pairs[pair] = persona_pairs.get(pair, 0) + 1
        
        # Count scenarios
        scenarios = {}
        for dialogue in dialogues:
            scenarios[dialogue.scenario] = scenarios.get(dialogue.scenario, 0) + 1
        
        # Average consistency
        avg_consistency = sum(d.consistency_score for d in dialogues) / len(dialogues)
        
        return {
            "total_dialogues": len(dialogues),
            "persona_pairs": persona_pairs,
            "scenarios": scenarios,
            "avg_consistency_score": avg_consistency,
            "file_path": str(self.training_file)
        }
    
    def convert_to_training_format(self, dialogues: Optional[List[DualPersonaDialogue]] = None) -> List[Dict]:
        """Convert dual persona dialogues to standard training format"""
        if dialogues is None:
            dialogues = self.load_dialogues()
        
        training_data = [dialogue.to_training_format() for dialogue in dialogues]
        logger.info(f"Converted {len(training_data)} dual persona dialogues to training format")
        return training_data
    
    def check_data_exists(self) -> bool:
        """Check if dual persona data exists"""
        return self.training_file.exists()


def load_dual_persona_training_data(pipeline_dir: Optional[str] = None) -> List[Dict]:
    """
    Convenience function to load dual persona training data
    
    Args:
        pipeline_dir: Optional path to dual persona pipeline directory
        
    Returns:
        List of training examples in standard format
    """
    loader = DualPersonaLoader(pipeline_dir) if pipeline_dir else DualPersonaLoader()
    return loader.convert_to_training_format()


if __name__ == "__main__":
    # Test the loader
    loader = DualPersonaLoader()
    
    print("Dual Persona Training Data Loader")
    print("=" * 60)
    
    # Load and show statistics
    stats = loader.get_statistics()
    print(f"\n📊 Statistics:")
    print(f"   Total dialogues: {stats['total_dialogues']}")
    print(f"   Avg consistency: {stats['avg_consistency_score']:.2f}")
    
    if stats['persona_pairs']:
        print(f"\n👥 Persona Pairs:")
        for pair, count in stats['persona_pairs'].items():
            print(f"   {pair}: {count}")
    
    if stats['scenarios']:
        print(f"\n📝 Scenarios:")
        for scenario, count in list(stats['scenarios'].items())[:5]:
            print(f"   {scenario}: {count}")
    
    # Load training data
    training_data = loader.convert_to_training_format()
    print(f"\n✅ Loaded {len(training_data)} training examples")
    
    if training_data:
        print(f"\n📝 Sample example:")
        sample = training_data[0]
        print(f"   Personas: {sample['metadata']['persona_1']} + {sample['metadata']['persona_2']}")
        print(f"   Scenario: {sample['metadata']['scenario']}")
        print(f"   Text: {sample['text'][:200]}...")
