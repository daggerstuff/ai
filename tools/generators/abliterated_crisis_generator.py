#!/usr/bin/env python3
"""
Crisis Conversation Generator using Abliterated Models
Generates authentic crisis intervention training data using models without safety filters.
"""

import json
import requests
import time
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CrisisScenario:
    """Defines a crisis scenario for conversation generation"""
    name: str
    description: str
    intensity_level: int  # 1-10 scale
    demographics: Dict[str, str]
    crisis_indicators: List[str]
    expected_duration: int  # minutes
    therapeutic_goals: List[str]

class AbliteratedCrisisGenerator:
    """Generates crisis conversations using abliterated models"""
    
    def __init__(self, api_base_url: str = "https://api.pixelatedempathy.tech"):
        self.api_base_url = api_base_url
        self.model_name = "huihui_ai/qwen3-abliterated:4b-thinking-2507-q4_K_M"
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # Crisis scenarios for training data generation
        self.crisis_scenarios = self._initialize_crisis_scenarios()
        
    def _initialize_crisis_scenarios(self) -> List[CrisisScenario]:
        """Initialize comprehensive crisis scenarios"""
        return [
            CrisisScenario(
                name="Acute Suicidal Ideation",
                description="Individual expressing active suicidal thoughts with plan",
                intensity_level=10,
                demographics={"age": "25-35", "gender": "any", "background": "recent job loss"},
                crisis_indicators=["suicidal ideation", "hopelessness", "isolation", "plan formation"],
                expected_duration=45,
                therapeutic_goals=["safety planning", "crisis de-escalation", "resource connection"]
            ),
            CrisisScenario(
                name="Self-Harm Crisis",
                description="Person engaging in self-injurious behavior as coping mechanism",
                intensity_level=8,
                demographics={"age": "16-24", "gender": "any", "background": "academic pressure"},
                crisis_indicators=["self-harm", "emotional dysregulation", "shame", "secrecy"],
                expected_duration=30,
                therapeutic_goals=["harm reduction", "coping skills", "emotional validation"]
            ),
            CrisisScenario(
                name="Severe Depression Episode",
                description="Major depressive episode with functional impairment",
                intensity_level=7,
                demographics={"age": "30-50", "gender": "any", "background": "chronic illness"},
                crisis_indicators=["anhedonia", "fatigue", "worthlessness", "cognitive impairment"],
                expected_duration=40,
                therapeutic_goals=["mood stabilization", "activity scheduling", "cognitive restructuring"]
            ),
            CrisisScenario(
                name="Domestic Violence Crisis",
                description="Survivor of domestic violence seeking immediate help",
                intensity_level=9,
                demographics={"age": "25-45", "gender": "any", "background": "financial dependence"},
                crisis_indicators=["fear", "trauma symptoms", "safety concerns", "isolation"],
                expected_duration=50,
                therapeutic_goals=["safety planning", "resource connection", "empowerment"]
            ),
            CrisisScenario(
                name="LGBTQ+ Identity Crisis",
                description="Individual struggling with identity acceptance and family rejection",
                intensity_level=6,
                demographics={"age": "16-25", "gender": "LGBTQ+", "background": "religious family"},
                crisis_indicators=["identity confusion", "family conflict", "social rejection", "depression"],
                expected_duration=35,
                therapeutic_goals=["identity affirmation", "family mediation", "community connection"]
            ),
            CrisisScenario(
                name="Substance Abuse Crisis",
                description="Person in active addiction seeking help after overdose scare",
                intensity_level=8,
                demographics={"age": "20-40", "gender": "any", "background": "unemployment"},
                crisis_indicators=["addiction", "health complications", "legal issues", "relationship breakdown"],
                expected_duration=45,
                therapeutic_goals=["detox planning", "treatment referral", "harm reduction"]
            ),
            CrisisScenario(
                name="Grief and Loss Crisis",
                description="Individual overwhelmed by sudden loss of loved one",
                intensity_level=7,
                demographics={"age": "35-65", "gender": "any", "background": "sudden death of spouse"},
                crisis_indicators=["acute grief", "disbelief", "anger", "functional impairment"],
                expected_duration=40,
                therapeutic_goals=["grief processing", "support system activation", "coping strategies"]
            ),
            CrisisScenario(
                name="Panic Disorder Crisis",
                description="Person experiencing severe panic attacks with agoraphobia",
                intensity_level=6,
                demographics={"age": "20-40", "gender": "any", "background": "work stress"},
                crisis_indicators=["panic attacks", "avoidance", "physical symptoms", "catastrophic thinking"],
                expected_duration=25,
                therapeutic_goals=["panic management", "breathing techniques", "exposure planning"]
            ),
            CrisisScenario(
                name="Psychotic Episode",
                description="Individual experiencing first psychotic break",
                intensity_level=9,
                demographics={"age": "18-25", "gender": "any", "background": "college student"},
                crisis_indicators=["hallucinations", "delusions", "disorganized thinking", "agitation"],
                expected_duration=60,
                therapeutic_goals=["reality testing", "medication evaluation", "family support"]
            ),
            CrisisScenario(
                name="Eating Disorder Crisis",
                description="Person with severe eating disorder at medical risk",
                intensity_level=8,
                demographics={"age": "16-30", "gender": "any", "background": "perfectionism"},
                crisis_indicators=["restrictive eating", "body dysmorphia", "medical complications", "secrecy"],
                expected_duration=45,
                therapeutic_goals=["medical stabilization", "nutritional counseling", "body image work"]
            ),
            CrisisScenario(
                name="Veteran PTSD Crisis",
                description="Military veteran with combat PTSD having flashbacks",
                intensity_level=8,
                demographics={"age": "25-45", "gender": "any", "background": "combat veteran"},
                crisis_indicators=["flashbacks", "hypervigilance", "nightmares", "emotional numbing"],
                expected_duration=50,
                therapeutic_goals=["grounding techniques", "trauma processing", "veteran resources"]
            ),
            CrisisScenario(
                name="Adolescent Crisis",
                description="Teenager with behavioral issues and family conflict",
                intensity_level=6,
                demographics={"age": "13-17", "gender": "any", "background": "divorced parents"},
                crisis_indicators=["behavioral problems", "family conflict", "school issues", "peer pressure"],
                expected_duration=35,
                therapeutic_goals=["family therapy", "behavioral interventions", "peer support"]
            ),
            CrisisScenario(
                name="Elderly Depression Crisis",
                description="Older adult with late-onset depression and isolation",
                intensity_level=7,
                demographics={"age": "65+", "gender": "any", "background": "recent retirement"},
                crisis_indicators=["late-onset depression", "social isolation", "health decline", "hopelessness"],
                expected_duration=40,
                therapeutic_goals=["social connection", "activity engagement", "medical coordination"]
            ),
            CrisisScenario(
                name="Workplace Trauma Crisis",
                description="Employee experiencing workplace harassment and trauma",
                intensity_level=7,
                demographics={"age": "25-55", "gender": "any", "background": "corporate environment"},
                crisis_indicators=["workplace trauma", "anxiety", "sleep disturbance", "avoidance"],
                expected_duration=35,
                therapeutic_goals=["trauma processing", "workplace advocacy", "stress management"]
            ),
            CrisisScenario(
                name="Postpartum Crisis",
                description="New mother with severe postpartum depression and anxiety",
                intensity_level=8,
                demographics={"age": "20-40", "gender": "female", "background": "first-time mother"},
                crisis_indicators=["postpartum depression", "anxiety", "bonding difficulties", "guilt"],
                expected_duration=45,
                therapeutic_goals=["maternal mental health", "bonding support", "family resources"]
            )
        ]
    
    def generate_crisis_conversation(self, scenario: CrisisScenario, num_exchanges: int = 10) -> Dict:
        """Generate a crisis conversation based on scenario"""
        logger.info(f"Generating crisis conversation for scenario: {scenario.name}")
        
        # Create system prompt for crisis conversation
        system_prompt = self._create_system_prompt(scenario)
        
        conversation = {
            "scenario": scenario.name,
            "metadata": {
                "intensity_level": scenario.intensity_level,
                "demographics": scenario.demographics,
                "crisis_indicators": scenario.crisis_indicators,
                "therapeutic_goals": scenario.therapeutic_goals,
                "generated_at": datetime.now().isoformat(),
                "model_used": self.model_name
            },
            "conversation": []
        }
        
        # Generate conversation exchanges
        context_messages = [{"role": "system", "content": system_prompt}]
        
        for exchange in range(num_exchanges):
            try:
                # Generate client message
                client_prompt = self._create_client_prompt(scenario, exchange, conversation["conversation"])
                client_response = self._call_model(context_messages + [{"role": "user", "content": client_prompt}])
                
                if client_response:
                    conversation["conversation"].append({
                        "speaker": "client",
                        "message": client_response,
                        "exchange_number": exchange + 1,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    # Generate counselor response
                    counselor_prompt = self._create_counselor_prompt(scenario, client_response)
                    counselor_response = self._call_model(context_messages + [
                        {"role": "user", "content": f"Client said: {client_response}"},
                        {"role": "user", "content": counselor_prompt}
                    ])
                    
                    if counselor_response:
                        conversation["conversation"].append({
                            "speaker": "counselor",
                            "message": counselor_response,
                            "exchange_number": exchange + 1,
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Brief pause between exchanges
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error generating exchange {exchange + 1}: {e}")
                break
        
        # Add quality assessment
        conversation["quality_assessment"] = self._assess_conversation_quality(conversation)
        
        return conversation
    
    def _create_system_prompt(self, scenario: CrisisScenario) -> str:
        """Create system prompt for crisis conversation generation"""
        return f"""You are generating realistic crisis intervention training data. Create authentic dialogue that demonstrates both the client's crisis state and appropriate therapeutic responses.

SCENARIO: {scenario.name}
DESCRIPTION: {scenario.description}
INTENSITY: {scenario.intensity_level}/10
DEMOGRAPHICS: {scenario.demographics}
CRISIS INDICATORS: {', '.join(scenario.crisis_indicators)}
THERAPEUTIC GOALS: {', '.join(scenario.therapeutic_goals)}

Generate realistic, authentic dialogue that:
1. Shows genuine crisis presentation with appropriate emotional intensity
2. Includes realistic language patterns and expressions of distress
3. Demonstrates crisis intervention techniques and therapeutic responses
4. Maintains clinical authenticity while being educational
5. Shows progression through crisis intervention stages

This is for training crisis counselors to recognize and respond to real crisis situations."""
    
    def _create_client_prompt(self, scenario: CrisisScenario, exchange: int, conversation_history: List) -> str:
        """Create prompt for client message generation"""
        stage = "initial" if exchange < 3 else "middle" if exchange < 7 else "resolution"
        
        return f"""Generate a realistic client message for a {scenario.name} crisis scenario.
Exchange #{exchange + 1} - {stage} stage
Intensity level: {scenario.intensity_level}/10
Demographics: {scenario.demographics}

The client should express:
- Authentic emotional distress appropriate to the crisis
- Language patterns typical of someone in crisis
- Specific concerns related to {', '.join(scenario.crisis_indicators)}
- Responses that show the crisis intervention process

Make this realistic and authentic for training purposes. The client is in genuine distress and needs help."""
    
    def _create_counselor_prompt(self, scenario: CrisisScenario, client_message: str) -> str:
        """Create prompt for counselor response generation"""
        return f"""Generate a professional crisis counselor response to this client message: "{client_message}"

The counselor should demonstrate:
- Crisis intervention techniques appropriate for {scenario.name}
- Empathetic and validating language
- Safety assessment and planning
- Therapeutic goals: {', '.join(scenario.therapeutic_goals)}
- Professional boundaries and ethical practice

Provide a realistic counselor response that shows best practices in crisis intervention."""
    
    def _call_model(self, messages: List[Dict], max_tokens: int = 200) -> Optional[str]:
        """Call the abliterated model API"""
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.8,
                "top_p": 0.9
            }
            
            response = self.session.post(
                f"{self.api_base_url}/v1/chat/completions",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    # Clean up thinking tags if present
                    if "<think>" in content:
                        content = content.split("</think>")[-1].strip()
                    return content
                else:
                    logger.error(f"No choices in response: {result}")
                    return None
            else:
                logger.error(f"API call failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling model: {e}")
            return None
    
    def _assess_conversation_quality(self, conversation: Dict) -> Dict:
        """Assess the quality of generated conversation"""
        assessment = {
            "authenticity_score": 0,
            "therapeutic_quality": 0,
            "crisis_indicators_present": [],
            "therapeutic_techniques_used": [],
            "areas_for_improvement": []
        }
        
        # Analyze conversation content
        all_text = " ".join([msg["message"] for msg in conversation["conversation"]])
        
        # Check for crisis indicators
        crisis_keywords = ["suicide", "kill", "hurt", "pain", "hopeless", "worthless", "scared", "help"]
        for keyword in crisis_keywords:
            if keyword.lower() in all_text.lower():
                assessment["crisis_indicators_present"].append(keyword)
        
        # Check for therapeutic techniques
        therapeutic_keywords = ["understand", "feel", "safe", "support", "help", "together", "plan"]
        for keyword in therapeutic_keywords:
            if keyword.lower() in all_text.lower():
                assessment["therapeutic_techniques_used"].append(keyword)
        
        # Calculate scores (simplified)
        assessment["authenticity_score"] = min(10, len(assessment["crisis_indicators_present"]) * 2)
        assessment["therapeutic_quality"] = min(10, len(assessment["therapeutic_techniques_used"]) * 1.5)
        
        return assessment
    
    def generate_training_dataset(self, num_conversations_per_scenario: int = 2) -> List[Dict]:
        """Generate a complete training dataset"""
        logger.info(f"Generating training dataset with {num_conversations_per_scenario} conversations per scenario")
        
        dataset = []
        
        for scenario in self.crisis_scenarios:
            for i in range(num_conversations_per_scenario):
                logger.info(f"Generating conversation {i+1}/{num_conversations_per_scenario} for {scenario.name}")
                
                conversation = self.generate_crisis_conversation(scenario)
                if conversation:
                    dataset.append(conversation)
                
                # Pause between conversations to avoid overwhelming the API
                time.sleep(5)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str = None) -> str:
        """Save generated dataset to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crisis_training_dataset_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset saved to {filename}")
        return filename

def main():
    """Main function to generate crisis training data"""
    generator = AbliteratedCrisisGenerator()
    
    # Test with a single scenario first
    test_scenario = generator.crisis_scenarios[0]  # Acute Suicidal Ideation
    logger.info("Testing with single scenario...")
    
    conversation = generator.generate_crisis_conversation(test_scenario, num_exchanges=5)
    
    if conversation:
        print("\n" + "="*50)
        print(f"GENERATED CONVERSATION: {conversation['scenario']}")
        print("="*50)
        
        for msg in conversation['conversation']:
            speaker = msg['speaker'].upper()
            message = msg['message']
            print(f"\n{speaker}: {message}")
        
        print("\n" + "="*50)
        print("QUALITY ASSESSMENT:")
        print(json.dumps(conversation['quality_assessment'], indent=2))
        print("="*50)
        
        # Save single conversation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_crisis_conversation_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
        print(f"\nConversation saved to {filename}")
    else:
        print("Failed to generate conversation")

if __name__ == "__main__":
    main()
