#!/usr/bin/env python3
"""
Standalone Edge Case Generator for Pixelated Empathy
Combines prompt generation, formatting, and synthesis in one pipeline
"""

import json
import re
import time
import random
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import pandas as pd
from tqdm import tqdm


@dataclass
class EdgeCaseScenario:
    """Structure for edge case scenarios"""

    category: str
    scenario_id: str
    description: str
    template: str
    difficulty_level: str
    expected_challenges: List[str]


class EdgeCaseGenerator:
    """Complete edge case generation pipeline"""

    def __init__(
        self,
        api_provider: str = "ollama",
        api_key: Optional[str] = None,
        model_name: str = "artifish/llama3.2-uncensored",
        output_dir: str = "output",
    ):

        self.api_provider = api_provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Setup API client
        self._setup_api_client()

        # Define edge case categories and templates
        self.edge_case_categories = self._define_edge_case_categories()

    def _setup_api_client(self):
        """Setup API client based on provider"""
        if self.api_provider == "openai":
            try:
                import openai

                self.client = openai.OpenAI(api_key=self.api_key)
            except ImportError as e:
                raise ImportError("OpenAI package not installed. Run: pip install openai") from e

        elif self.api_provider == "anthropic":
            try:
                import anthropic

                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError as e:
                raise ImportError(
                    "Anthropic package not installed. Run: pip install anthropic"
                ) from e

        elif self.api_provider == "ollama":
            # For local Ollama - no API key needed
            self.base_url = "http://localhost:11434"

        else:
            raise ValueError(f"Unsupported API provider: {self.api_provider}")

    def _define_edge_case_categories(self) -> Dict[str, Dict]:
        """Define the 25 edge case categories with templates"""
        return {
            "suicidality": {
                "description": "Clients presenting with suicidal ideation, plans, or attempts",
                "template": "Simulate a therapy session set in a tense, dimly lit room. Bob is a 35-year-old man who has been struggling with severe depression. He expresses thoughts of ending his life.",
                "difficulty": "very_high",
                "challenges": ["crisis_intervention", "safety_assessment", "resource_provision"],
            },
            "homicidal_client": {
                "description": "Clients expressing violent thoughts toward others",
                "template": "Simulate a therapy session with Sarah, a 28-year-old woman who has been having violent fantasies about harming her coworkers.",
                "difficulty": "very_high",
                "challenges": ["duty_to_warn", "safety_planning", "legal_considerations"],
            },
            "substance_abuse_crisis": {
                "description": "Clients in active addiction crisis or withdrawal",
                "template": "Create a therapy session with Mike, a 42-year-old who shows up to session clearly intoxicated and becoming increasingly agitated.",
                "difficulty": "high",
                "challenges": ["immediate_safety", "medical_concerns", "boundary_setting"],
            },
            "psychotic_episode": {
                "description": "Clients experiencing active psychosis or delusions",
                "template": "Simulate a session with Lisa, a 24-year-old who believes the government is monitoring her through the therapy cameras.",
                "difficulty": "very_high",
                "challenges": ["reality_testing", "therapeutic_alliance", "psychiatric_emergency"],
            },
            "severe_dissociation": {
                "description": "Clients with dissociative identity disorder or severe dissociative episodes",
                "template": "Create a therapy session where the client suddenly switches personalities mid-conversation and doesn't remember the previous discussion.",
                "difficulty": "high",
                "challenges": ["continuity_of_care", "identity_validation", "grounding_techniques"],
            },
            "manic_episode": {
                "description": "Clients in manic or hypomanic states",
                "template": "Simulate a session with David, who speaks rapidly, jumps between topics, and believes he has special powers.",
                "difficulty": "high",
                "challenges": ["medication_compliance", "reality_orientation", "energy_management"],
            },
            "trauma_flashback": {
                "description": "Clients experiencing active trauma flashbacks",
                "template": "Create a session where the client suddenly becomes triggered and starts reliving a traumatic experience.",
                "difficulty": "high",
                "challenges": ["grounding", "safety", "present_moment_awareness"],
            },
            "eating_disorder_medical": {
                "description": "Clients with severe eating disorders at medical risk",
                "template": "Simulate a session with Alex, severely underweight and recently discharged from medical hospitalization for anorexia.",
                "difficulty": "high",
                "challenges": ["medical_monitoring", "food_fear", "body_image_distortion"],
            },
            "borderline_crisis": {
                "description": "Clients with BPD in emotional crisis",
                "template": "Create a therapy session with Jordan, who threatens self-harm after feeling abandoned by the therapist's vacation.",
                "difficulty": "high",
                "challenges": ["emotional_regulation", "abandonment_fears", "self_harm_prevention"],
            },
            "paranoid_accusations": {
                "description": "Clients making paranoid accusations against therapist",
                "template": "Simulate a session where the client accuses the therapist of plotting against them with their family.",
                "difficulty": "high",
                "challenges": ["therapeutic_alliance", "boundary_maintenance", "trust_building"],
            },
            "sexual_trauma_disclosure": {
                "description": "Clients disclosing sexual trauma for the first time",
                "template": "Create a session where the client breaks down while revealing childhood sexual abuse.",
                "difficulty": "high",
                "challenges": [
                    "trauma_sensitive_response",
                    "reporting_requirements",
                    "safety_planning",
                ],
            },
            "child_abuse_reporting": {
                "description": "Situations requiring mandated reporting for child abuse",
                "template": "Simulate a session where a parent admits to physically harming their child.",
                "difficulty": "very_high",
                "challenges": ["mandated_reporting", "therapeutic_alliance", "child_safety"],
            },
            "elder_abuse_concerns": {
                "description": "Situations involving elder abuse or neglect",
                "template": "Create a session with an elderly client showing signs of financial and physical abuse by caregivers.",
                "difficulty": "high",
                "challenges": [
                    "vulnerability_assessment",
                    "reporting_decisions",
                    "resource_coordination",
                ],
            },
            "domestic_violence_active": {
                "description": "Clients in active domestic violence situations",
                "template": "Simulate a session with someone who arrives with fresh bruises and minimizes their partner's violence.",
                "difficulty": "high",
                "challenges": ["safety_planning", "ambivalence", "lethality_assessment"],
            },
            "stalking_harassment": {
                "description": "Clients being stalked or harassed",
                "template": "Create a session with a client who reports being followed and receiving threatening messages.",
                "difficulty": "moderate",
                "challenges": ["safety_measures", "documentation", "legal_resources"],
            },
            "religious_delusions": {
                "description": "Clients with religious or spiritual delusions",
                "template": "Simulate a session with someone who believes they are receiving direct messages from God to hurt others.",
                "difficulty": "high",
                "challenges": ["cultural_sensitivity", "delusion_vs_faith", "safety_assessment"],
            },
            "medication_refusal": {
                "description": "Clients refusing essential psychiatric medications",
                "template": "Create a session with a bipolar client who stops taking lithium because they miss feeling 'creative.'",
                "difficulty": "moderate",
                "challenges": ["medication_adherence", "autonomy_vs_safety", "psychoeducation"],
            },
            "family_therapy_conflict": {
                "description": "Explosive conflicts in family therapy sessions",
                "template": "Simulate a family session where parents start screaming at each other and threatening divorce.",
                "difficulty": "moderate",
                "challenges": ["de_escalation", "session_management", "safety_for_children"],
            },
            "adolescent_defiance": {
                "description": "Extremely defiant or aggressive adolescent clients",
                "template": "Create a session with a 16-year-old who refuses to speak, throws objects, and threatens to leave.",
                "difficulty": "moderate",
                "challenges": ["engagement", "boundary_setting", "family_dynamics"],
            },
            "couple_therapy_betrayal": {
                "description": "Couples therapy with major betrayals revealed",
                "template": "Simulate a couples session where one partner reveals an ongoing affair and secret financial debt.",
                "difficulty": "moderate",
                "challenges": ["emotional_safety", "neutrality", "relationship_assessment"],
            },
            "grief_complicated": {
                "description": "Clients with complicated or traumatic grief",
                "template": "Create a session with someone whose child died by suicide and they blame themselves completely.",
                "difficulty": "high",
                "challenges": ["guilt_processing", "meaning_making", "suicide_risk"],
            },
            "cultural_conflicts": {
                "description": "Intense cultural or religious value conflicts",
                "template": "Simulate a session with a client torn between family cultural expectations and personal identity.",
                "difficulty": "moderate",
                "challenges": ["cultural_competence", "identity_exploration", "family_dynamics"],
            },
            "therapist_boundaries": {
                "description": "Clients testing or violating therapeutic boundaries",
                "template": "Create a session where the client asks for personal information and suggests meeting outside of therapy.",
                "difficulty": "moderate",
                "challenges": [
                    "boundary_maintenance",
                    "therapeutic_frame",
                    "exploitation_prevention",
                ],
            },
            "therapy_resistance": {
                "description": "Clients showing extreme resistance to therapeutic process",
                "template": "Simulate a session with someone court-ordered to therapy who insists they don't need help.",
                "difficulty": "moderate",
                "challenges": ["motivation", "engagement", "involuntary_treatment"],
            },
            "chronic_pain_despair": {
                "description": "Clients with chronic pain considering ending treatment",
                "template": "Create a session with someone in chronic pain who wants to stop all medical treatment and 'let nature take its course.'",
                "difficulty": "moderate",
                "challenges": ["pain_psychology", "medical_collaboration", "quality_of_life"],
            },
        }

    def _write_jsonl_file(self, filepath: Path, items: List[Dict]):
        """Write a list of dicts to a JSONL file."""
        with open(filepath, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    def generate_prompts(self, scenarios_per_category: int = 20) -> List[Dict]:
        """Generate prompts for all edge case scenarios"""
        print("Generating edge case prompts...")
        all_prompts = []
        prompt_id = 1
        for category, details in self.edge_case_categories.items():
            print(f"Generating {scenarios_per_category} prompts for {category}...")
            for i in range(1, scenarios_per_category + 1):
                scenario_id = f"{category}_{i:03d}"
                # Create variations of the base template
                template = details["template"]
                if i > 1:
                    # Add variations for subsequent prompts
                    variations = self._create_template_variations(template, i)
                    template = random.choice(variations)
                prompt_data = {
                    "prompt_id": f"edge_case_{prompt_id:04d}",
                    "scenario_id": scenario_id,
                    "category": category,
                    "difficulty_level": details["difficulty"],
                    "expected_challenges": details["challenges"],
                    "instructions": template,
                    "created_at": datetime.now().isoformat(),
                }
                all_prompts.append(prompt_data)
                prompt_id += 1
        return self._extracted_from_create_training_format_28(
            "edge_case_prompts.jsonl",
            all_prompts,
            "Generated ",
            " prompts and saved to ",
        )

    def _create_template_variations(self, base_template: str, variation_num: int) -> List[str]:
        """Create variations of base templates"""
        variations = [base_template]
        # Add age variations
        age_patterns = [
            ("35-year-old", f"{30 + (variation_num % 20)}-year-old"),
            ("28-year-old", f"{25 + (variation_num % 15)}-year-old"),
            ("42-year-old", f"{35 + (variation_num % 25)}-year-old"),
            ("24-year-old", f"{20 + (variation_num % 20)}-year-old"),
            ("16-year-old", f"{14 + (variation_num % 6)}-year-old"),
        ]
        # Add name variations
        names = ["Alex", "Jordan", "Casey", "Riley", "Taylor", "Morgan", "Avery", "Quinn"]
        variations.extend(
            base_template.replace(original_age, new_age)
            for original_age, new_age in age_patterns
            if original_age in base_template
        )
        # Add setting variations
        if "dimly lit room" in base_template:
            settings = [
                "bright office",
                "cozy therapy room",
                "clinical setting",
                "comfortable space",
            ]
            variations.extend(
                [
                    base_template.replace("dimly lit room", setting)
                    for setting in settings[: variation_num % len(settings)]
                ]
            )
        return variations

    def generate_conversations(
        self, prompts: List[Dict], max_conversations: int = None
    ) -> List[Dict]:
        """Generate conversations from prompts using selected API"""
        if max_conversations:
            prompts = prompts[:max_conversations]
        print(f"Generating {len(prompts)} conversations using {self.api_provider}...")
        conversations = []
        failed_prompts = []
        progress_bar = tqdm(prompts, desc="Generating conversations")
        for prompt_data in progress_bar:
            try:
                if conversation := self._generate_single_conversation(prompt_data):
                    conversations.append(conversation)
                    # Save progress every 10 conversations
                    if len(conversations) % 10 == 0:
                        self._save_progress(conversations, failed_prompts)
                else:
                    failed_prompts.append(prompt_data)
            except Exception as e:
                print(f"Error generating conversation for {prompt_data['scenario_id']}: {e}")
                failed_prompts.append(prompt_data)
            # Add delay to avoid rate limiting
            time.sleep(1)
        # Final save
        self._save_progress(conversations, failed_prompts)
        print(f"Generated {len(conversations)} conversations")
        print(f"Failed prompts: {len(failed_prompts)}")
        return conversations

    def _generate_single_conversation(self, prompt_data: Dict) -> Optional[Dict]:
        """Generate a single conversation from prompt"""
        system_prompt = """You are a difficult therapy client simulator. Generate a realistic, challenging therapy dialogue that will help train therapists to handle difficult situations.

Create a conversation with:
1. Therapist statements that are professional and appropriate
2. Client responses that demonstrate the specific challenging behavior described
3. Realistic emotional dynamics
4. Educational value for therapist training

Format the response as a dialogue with clear speaker labels.
Make the client responses authentic and challenging without being harmful or unethical."""
        user_prompt = f"""Generate a therapy dialogue based on this scenario:

{prompt_data['instructions']}

Category: {prompt_data['category']}
Difficulty: {prompt_data['difficulty_level']}
Expected challenges: {', '.join(prompt_data['expected_challenges'])}

Create a realistic dialogue between therapist and client that demonstrates these challenges."""
        try:
            if self.api_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=1000,
                    temperature=0.7,
                )
                generated_text = response.choices[0].message.content
            elif self.api_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                generated_text = response.content[0].text
            elif self.api_provider == "ollama":
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": f"{system_prompt}\n\n{user_prompt}",
                        "stream": False,
                    },
                    timeout=120,
                )
                if response.status_code == 200:
                    generated_text = response.json()["response"]
                else:
                    return None
            # Extract Q&A pairs from generated text
            if qa_pairs := self._extract_qa_pairs(generated_text):
                return {
                    **prompt_data,
                    "generated_text": generated_text,
                    "qa_pairs": qa_pairs,
                    "generated_at": datetime.now().isoformat(),
                }
        except Exception as e:
            print(f"API error: {e}")
            return None
        return None

    def _extract_qa_pairs(self, text: str) -> List[Dict]:
        """Extract Q&A pairs from generated dialogue"""
        lines = text.strip().split("\n")
        pairs = []
        current_therapist = ""
        current_client = ""
        for line in lines:
            if not (stripped_line := line.strip()):
                continue
            # Look for therapist/client indicators
            if any(
                indicator in stripped_line.lower()
                for indicator in ["therapist:", "therapy:", "counselor:", "dr.", "psychologist:"]
            ):
                if current_therapist and current_client:
                    pairs.append(
                        {
                            "prompt": current_therapist.strip(),
                            "response": current_client.strip(),
                            "speaker_roles": {"prompt": "therapist", "response": "client"},
                        }
                    )
                    current_client = ""
                current_therapist = re.sub(r"^[^:]*:\s*", "", stripped_line)
            elif any(
                indicator in stripped_line.lower()
                for indicator in ["client:", "patient:", "person:"]
            ):
                current_client = re.sub(r"^[^:]*:\s*", "", stripped_line)
            elif current_therapist and not current_client:
                current_therapist += " " + stripped_line
            elif current_client:
                current_client += " " + stripped_line
        # Add final pair if exists
        if current_therapist and current_client:
            pairs.append(
                {
                    "prompt": current_therapist.strip(),
                    "response": current_client.strip(),
                    "speaker_roles": {"prompt": "therapist", "response": "client"},
                }
            )
        return pairs

    def _save_progress(self, conversations: List[Dict], failed_prompts: List[Dict]):
        """Save progress to files"""
        # Save successful conversations
        conversations_file = self.output_dir / "generated_conversations.jsonl"
        self._write_jsonl_file(conversations_file, conversations)
        # Save failed prompts for retry
        if failed_prompts:
            failed_file = self.output_dir / "failed_prompts.jsonl"
            self._write_jsonl_file(failed_file, failed_prompts)

    def create_training_format(self, conversations: List[Dict]) -> List[Dict]:
        """Convert conversations to training format"""
        training_data = []
        for conv in conversations:
            if "qa_pairs" in conv:
                for qa_pair in conv["qa_pairs"]:
                    training_item = {
                        "prompt": qa_pair["prompt"],
                        "response": qa_pair["response"],
                        "purpose": "difficult_client",
                        "category": conv["category"],
                        "difficulty_level": conv["difficulty_level"],
                        "expected_challenges": conv["expected_challenges"],
                        "source": "edge_case_generation",
                        "generated_at": conv["generated_at"],
                    }
                    training_data.append(training_item)
        return self._extracted_from_create_training_format_28(
            "edge_cases_training_format.jsonl",
            training_data,
            "Created ",
            " training examples in ",
        )

    # TODO Rename this here and in `generate_prompts` and `create_training_format`
    def _extracted_from_create_training_format_28(self, arg0, arg1, arg2, arg3):
        prompts_file = self.output_dir / arg0
        self._write_jsonl_file(prompts_file, arg1)
        print(f"{arg2}{len(arg1)}{arg3}{prompts_file}")
        return arg1

    def generate_summary_report(self, conversations: List[Dict]) -> str:
        """Generate summary report of edge case generation"""
        # Create statistics
        total_conversations = len(conversations)
        total_qa_pairs = sum(len(conv.get("qa_pairs", [])) for conv in conversations)
        category_counts = {}
        difficulty_counts = {}
        for conv in conversations:
            cat = conv.get("category", "unknown")
            diff = conv.get("difficulty_level", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1
        # Create report
        report = (
            "# Edge Case Generation Summary Report\n"
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            "## Overall Statistics\n"
            f"- Total Conversations Generated: {total_conversations}\n"
            f"- Total Q&A Pairs: {total_qa_pairs}\n"
            f"- Average Q&A Pairs per Conversation: {total_qa_pairs/max(total_conversations,1):.1f}\n\n"
            "## Category Breakdown\n"
        )
        for category, count in sorted(category_counts.items()):
            report += f"- {category}: {count} conversations\n"
        report += "\n## Difficulty Level Distribution\n"
        for difficulty, count in sorted(difficulty_counts.items()):
            report += f"- {difficulty}: {count} conversations\n"
        report += (
            "\n## Files Generated\n"
            "- Edge case prompts: edge_case_prompts.jsonl\n"
            "- Generated conversations: generated_conversations.jsonl  \n"
            "- Training format: edge_cases_training_format.jsonl\n"
            "- Summary report: summary_report.md\n\n"
            "## Next Steps\n"
            "1. Review generated conversations for quality\n"
            "2. Integrate with main training pipeline\n"
            "3. Evaluate model performance on edge cases\n"
        )
        # Save report
        report_file = self.output_dir / "summary_report.md"
        with open(report_file, "w") as f:
            f.write(report)
        return report


def main():
    """Example usage of EdgeCaseGenerator"""

    # Configuration
    generator = EdgeCaseGenerator(
        api_provider="ollama",  # Default: Ollama
        api_key=None,  # Ollama does not require an API key
        model_name="artifish/llama3.2-uncensored",
        output_dir="edge_case_output",
    )

    # Generate prompts (20 per category = 500 total)
    prompts = generator.generate_prompts(scenarios_per_category=20)

    # Generate conversations (start with smaller batch for testing)
    conversations = generator.generate_conversations(prompts, max_conversations=50)

    # Create training format
    training_data = generator.create_training_format(conversations)

    # Generate report
    report = generator.generate_summary_report(conversations)
    print(report)


if __name__ == "__main__":
    main()
