#!/usr/bin/env python3
"""
Edge Case Integrator - KAN-28 Component #3
Integrates nightmare fuel scenarios with expert voices and therapeutic frameworks
"""

import json
import logging
import random
import re
import uuid
from datetime import datetime, timezone
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

    def generate_crisis_and_cultural_edge_cases(
        self,
        *,
        target_records: int = 15_000,
        seed: int = 1,
        turns_per_scenario: int = 3,
        crisis_ratio: float = 0.5,
        output_file: str | None = None,
        summary_file: str | None = None,
        voice_blender: Any | None = None,
        bias_detector: Any | None = None,
    ) -> Dict[str, Any]:
        """Generate crisis + cultural edge case records and write them as JSONL.

        This is designed to create a stage-3 "edge stress test" dataset of therapist
        responses. Records are written to a location that can be uploaded to
        `s3://pixel-data/edge_cases/` via the existing OVH sync script.

        `crisis_ratio` must be between 0 and 1.
        """

        if target_records <= 0:
            raise ValueError("target_records must be > 0")

        if turns_per_scenario <= 0:
            raise ValueError("turns_per_scenario must be > 0")

        if not (0.0 <= crisis_ratio <= 1.0):
            raise ValueError("crisis_ratio must be between 0 and 1")

        rng = random.Random(seed)

        repo_root = Path(__file__).resolve().parents[2]
        default_output_dir = repo_root / "pipelines" / "edge_case_pipeline_standalone"
        output_file = output_file or str(
            default_output_dir / "edge_cases_crisis_cultural_15k.jsonl"
        )
        summary_file = summary_file or str(
            default_output_dir / "edge_cases_crisis_cultural_15k_summary.json"
        )

        total_scenarios = target_records // turns_per_scenario
        remainder = target_records % turns_per_scenario
        if remainder != 0:
            total_scenarios += 1

        crisis_scenarios = int(total_scenarios * crisis_ratio)
        cultural_scenarios = total_scenarios - crisis_scenarios

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_file).parent.mkdir(parents=True, exist_ok=True)

        counts = {
            "records_total": 0,
            "records_crisis": 0,
            "records_cultural": 0,
            "scenarios_total": total_scenarios,
            "scenarios_crisis": crisis_scenarios,
            "scenarios_cultural": cultural_scenarios,
            "stereotype_flags": 0,
            "bias_flags": 0,
        }

        failures: list[dict[str, Any]] = []

        with open(output_file, "w") as f:
            for scenario_index in range(total_scenarios):
                scenario_kind = "crisis" if scenario_index < crisis_scenarios else "cultural"

                if scenario_kind == "crisis":
                    scenario = self._generate_crisis_scenario(rng=rng)
                else:
                    scenario = self._generate_cultural_scenario(rng=rng)

                scenario_id = scenario["scenario_id"]
                turns: list[dict[str, Any]] = scenario["turns"]

                for turn_index, turn in enumerate(turns, start=1):
                    record = {
                        "id": str(uuid.uuid4()),
                        "scenario_id": scenario_id,
                        "scenario_kind": scenario_kind,
                        "scenario_subtype": scenario["scenario_subtype"],
                        "turn_index": turn_index,
                        "turns_total": len(turns),
                        "prompt": turn["client"],
                        "response": turn["therapist"],
                        "conversation": {
                            "client": turn["client"],
                            "therapist": turn["therapist"],
                        },
                        "metadata": {
                            "severity_level": scenario["severity_level"],
                            "evidence_based_tags": scenario["evidence_based_tags"],
                            "safety_triggers": turn.get("safety_triggers", []),
                            "cultural_context": scenario.get("cultural_context"),
                            "generated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    }

                    self._validate_record(record)

                    stereotype_flags = self._detect_harmful_stereotypes(
                        f"{record['prompt']}\n{record['response']}"
                    )
                    if stereotype_flags:
                        counts["stereotype_flags"] += 1
                        failures.append(
                            {
                                "scenario_id": scenario_id,
                                "id": record["id"],
                                "type": "stereotype",
                                "flags": stereotype_flags,
                            }
                        )

                    if bias_detector is not None:
                        bias_results = bias_detector.check_dataset_for_bias(record)
                        record["bias_detection"] = bias_results
                        if bias_results.get("overall_safety") != "safe":
                            counts["bias_flags"] += 1
                            failures.append(
                                {
                                    "scenario_id": scenario_id,
                                    "id": record["id"],
                                    "type": "bias_detection",
                                    "overall_safety": bias_results.get("overall_safety"),
                                }
                            )

                    if voice_blender is not None:
                        record["expert_voices"] = voice_blender.generate_tri_expert_responses(
                            record["prompt"],
                            context=scenario_kind,
                        )

                    f.write(json.dumps(record) + "\n")
                    counts["records_total"] += 1

                    if scenario_kind == "crisis":
                        counts["records_crisis"] += 1
                    else:
                        counts["records_cultural"] += 1

                    if counts["records_total"] >= target_records:
                        break

                if counts["records_total"] >= target_records:
                    break

        if counts["stereotype_flags"] != 0:
            raise ValueError(
                f"Generated dataset contains stereotype flags: {counts['stereotype_flags']}"
            )

        if counts["bias_flags"] != 0:
            raise ValueError(
                f"Generated dataset contains bias flags: {counts['bias_flags']}"
            )

        summary = {
            **counts,
            "output_file": output_file,
            "summary_file": summary_file,
            "seed": seed,
            "turns_per_scenario": turns_per_scenario,
            "crisis_ratio": crisis_ratio,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        if failures:
            # Failures should not happen due to the strict checks above, but keep
            # this in the summary for troubleshooting.
            summary["failures"] = failures[:50]

        return summary

    def _generate_crisis_scenario(self, *, rng: random.Random) -> dict[str, Any]:
        subtype = rng.choice(["suicidal_ideation", "panic", "severe_depression"])
        scenario_id = f"crisis_{subtype}_{uuid.uuid4().hex[:10]}"

        if subtype == "suicidal_ideation":
            turns = [
                {
                    "client": "I've been having thoughts about ending my life. I feel scared by how intense they are.",
                    "therapist": self._crisis_response(
                        "Thank you for telling me. I'm really glad you shared this with me. "
                        "Are you having thoughts of harming yourself right now, and are you alone? "
                        "If you are in immediate danger, call 911 (or your local emergency number). "
                        "In the U.S., you can also call or text 988 for the Suicide & Crisis Lifeline."
                    ),
                    "safety_triggers": ["risk_assessment", "988", "911"],
                },
                {
                    "client": "The thoughts keep coming back, and I'm worried I might act on them.",
                    "therapist": self._crisis_response(
                        "I'm hearing how serious and exhausting this is. Let's slow this down together. "
                        "Do you have a plan or access to anything you could use to hurt yourself? "
                        "If the urge feels immediate, call 911 (or your local emergency number) right now. "
                        "In the U.S., you can call or text 988, and we can also identify someone you trust to stay with you."
                    ),
                    "safety_triggers": ["means_check", "988", "911"],
                },
                {
                    "client": "I don't want to die, but I don't know how to keep myself safe tonight.",
                    "therapist": self._crisis_response(
                        "We can make a short-term safety plan for tonight. First: if you feel at risk of acting on these thoughts, "
                        "call 911 (or your local emergency number) or go to the nearest emergency department. "
                        "In the U.S., you can call or text 988. "
                        "Next, let's identify one person you can contact, and one way to reduce access to anything you could use to hurt yourself."
                    ),
                    "safety_triggers": ["safety_plan", "988", "911"],
                },
            ]
            severity_level = 9
            evidence_based_tags = ["crisis_risk_assessment", "safety_planning", "means_reduction"]

        elif subtype == "panic":
            turns = [
                {
                    "client": "I'm having a panic attack. My chest is tight and I feel like I can't breathe.",
                    "therapist": self._crisis_response(
                        "I'm here with you. Panic can feel terrifying, but it can pass. "
                        "If you think you may be having a medical emergency, call 911 (or your local emergency number). "
                        "If you're safe medically, let's try a grounding step: name 5 things you can see, 4 you can feel, 3 you can hear."
                    ),
                    "safety_triggers": ["medical_screen", "grounding", "911"],
                },
                {
                    "client": "I feel like I'm going to die and I can't calm down.",
                    "therapist": self._crisis_response(
                        "That fear makes sense. Let's work with your nervous system. "
                        "Breathe in for 4, hold for 2, out for 6, and repeat a few times. "
                        "If the symptoms worsen or you feel unsafe, call 911 (or your local emergency number). "
                        "In the U.S., 988 is also available if this panic is tied to feeling at risk of harming yourself."
                    ),
                    "safety_triggers": ["paced_breathing", "988", "911"],
                },
                {
                    "client": "I'm scared I'll hurt myself just to make it stop.",
                    "therapist": self._crisis_response(
                        "I'm really glad you said that out loud. Your safety comes first. "
                        "Are you having urges to harm yourself right now? "
                        "If there is any immediate risk, call 911 (or your local emergency number). "
                        "In the U.S., call or text 988. Let's also identify one person who can be with you while this wave passes."
                    ),
                    "safety_triggers": ["risk_assessment", "988", "911"],
                },
            ]
            severity_level = 8
            evidence_based_tags = ["panic_psychoeducation", "grounding", "paced_breathing"]

        else:
            turns = [
                {
                    "client": "I've been deeply depressed and can't get out of bed most days. Everything feels pointless.",
                    "therapist": self._crisis_response(
                        "I'm really sorry you're carrying that. Depression can make everything feel heavy and slow. "
                        "Before we go further, I want to check on safety: have you had thoughts about ending your life? "
                        "If you are in immediate danger, call 911 (or your local emergency number). In the U.S., you can also call or text 988."
                    ),
                    "safety_triggers": ["risk_assessment", "988", "911"],
                },
                {
                    "client": "I keep thinking people would be better off without me.",
                    "therapist": self._crisis_response(
                        "That thought is a common depression distortion, and it's also a sign we should take this seriously. "
                        "Do you have a plan, or access to anything you could use to hurt yourself? "
                        "If the risk feels immediate, call 911 (or your local emergency number). In the U.S., call or text 988."
                    ),
                    "safety_triggers": ["means_check", "988", "911"],
                },
                {
                    "client": "I want help, but I don't have energy to do anything.",
                    "therapist": self._crisis_response(
                        "We can keep this very small. One evidence-based approach is behavioral activation: "
                        "a tiny action can sometimes create a small shift in mood. "
                        "For today, what's one 2-minute step you can do, and who can you text to let them know you're struggling? "
                        "If you feel at risk of harming yourself, call 911 (or your local emergency number). In the U.S., call or text 988."
                    ),
                    "safety_triggers": ["behavioral_activation", "988", "911"],
                },
            ]
            severity_level = 7
            evidence_based_tags = ["suicide_risk_assessment", "behavioral_activation", "cognitive_restructuring"]

        return {
            "scenario_id": scenario_id,
            "scenario_subtype": subtype,
            "severity_level": severity_level,
            "evidence_based_tags": evidence_based_tags,
            "turns": turns,
        }

    def _generate_cultural_scenario(self, *, rng: random.Random) -> dict[str, Any]:
        subtype = rng.choice(["lgbtq_minority_stress", "religious_framework", "racial_discrimination"])
        scenario_id = f"cultural_{subtype}_{uuid.uuid4().hex[:10]}"

        if subtype == "lgbtq_minority_stress":
            turns = [
                {
                    "client": "I'm gay, and I'm afraid my family will reject me if they find out.",
                    "therapist": self._cultural_response(
                        "That fear makes a lot of sense, and your safety matters. "
                        "Do you want to explore what coming out would mean for you right now, or focus on coping with the stress? "
                        "We can talk through support options, boundaries, and what would help you feel safer."
                    ),
                    "safety_triggers": ["affirmation", "safety_planning"],
                },
                {
                    "client": "I feel ashamed, like I'm doing something wrong.",
                    "therapist": self._cultural_response(
                        "I'm hearing shame, and I want to name that shame often grows when we feel judged or unsafe. "
                        "Your identity isn't wrong. What messages did you learn about being gay, and how do they show up in your body today?"
                    ),
                    "safety_triggers": ["shame_resilience", "values_exploration"],
                },
                {
                    "client": "I want to tell my family, but I'm not sure it's safe.",
                    "therapist": self._cultural_response(
                        "We can make this decision based on your safety and your values. "
                        "Let's consider: what are the best-case and worst-case outcomes, what supports you can have in place, and what boundaries you might need. "
                        "If you ever feel at risk of harm or feel unsafe at home, it's okay to seek immediate help and support."
                    ),
                    "safety_triggers": ["risk_assessment", "support_network"],
                },
            ]
            severity_level = 6
            cultural_context = "lgbtq+ minority stress"
            evidence_based_tags = ["affirmative_therapy", "minority_stress", "shame_resilience"]

        elif subtype == "religious_framework":
            turns = [
                {
                    "client": "My faith is important to me, but I feel guilty all the time and I'm not sure if therapy fits.",
                    "therapist": self._cultural_response(
                        "Thank you for sharing that. Many people want care that respects their faith. "
                        "Would you like your spiritual beliefs to be part of our work together? "
                        "We can also explore how guilt shows up for you and whether it's aligned with your values or more about fear and self-judgment."
                    ),
                    "safety_triggers": ["cultural_humility", "values_alignment"],
                },
                {
                    "client": "My community says mental health struggles mean I'm not praying hard enough.",
                    "therapist": self._cultural_response(
                        "That message can feel heavy and isolating. Struggling doesn't mean you're failing spiritually. "
                        "We can hold both: your faith as a source of strength, and evidence-based mental health care as support. "
                        "What would compassionate support look like within your belief system?"
                    ),
                    "safety_triggers": ["psychoeducation", "self_compassion"],
                },
                {
                    "client": "I want to keep my faith, but I also want to feel better.",
                    "therapist": self._cultural_response(
                        "Those goals can fit together. We can explore coping practices that honor your faith (if you want), "
                        "alongside skills like grounding, cognitive reframing, and building social support. "
                        "What parts of your faith feel supportive, and what parts feel painful or pressured?"
                    ),
                    "safety_triggers": ["integrative_care", "cultural_humility"],
                },
            ]
            severity_level = 5
            cultural_context = "religious frameworks"
            evidence_based_tags = ["values_based", "cultural_humility", "integrative_care"]

        else:
            turns = [
                {
                    "client": "I'm facing discrimination at work, and it's affecting my sleep and mood.",
                    "therapist": self._cultural_response(
                        "I'm sorry you're dealing with that. Discrimination can have real mental and physical impacts. "
                        "When this happens, what do you notice in your body, and what supports are available to you at work or outside of work?"
                    ),
                    "safety_triggers": ["systemic_validation", "somatic_awareness"],
                },
                {
                    "client": "Sometimes I wonder if I'm overreacting, but it feels constant.",
                    "therapist": self._cultural_response(
                        "Questioning yourself is a common response when you're repeatedly invalidated. "
                        "We can explore what happened, how it affected you, and what boundaries or resources might help. "
                        "You don't have to carry this alone."
                    ),
                    "safety_triggers": ["validation", "support_network"],
                },
                {
                    "client": "I want to speak up, but I'm afraid of retaliation.",
                    "therapist": self._cultural_response(
                        "That fear makes sense. We can plan in a way that protects you: "
                        "clarify your goals, assess risk, and identify supports. "
                        "We can also focus on coping skills for stress responses while you decide what, if anything, to do next."
                    ),
                    "safety_triggers": ["risk_assessment", "problem_solving"],
                },
            ]
            severity_level = 6
            cultural_context = "minority stress / discrimination"
            evidence_based_tags = ["trauma_informed", "systemic_validation", "problem_solving"]

        return {
            "scenario_id": scenario_id,
            "scenario_subtype": subtype,
            "severity_level": severity_level,
            "evidence_based_tags": evidence_based_tags,
            "cultural_context": cultural_context,
            "turns": turns,
        }

    def _validate_record(self, record: dict[str, Any]) -> None:
        if not record.get("prompt") or not record.get("response"):
            raise ValueError("Record is missing prompt/response")

        scenario_kind = record.get("scenario_kind")
        if scenario_kind == "crisis":
            self._validate_crisis_response(record["response"])

    def _validate_crisis_response(self, response_text: str) -> None:
        required_terms = ["988", "911"]
        missing = [term for term in required_terms if term not in response_text]
        if missing:
            raise ValueError(f"Crisis response missing required safety triggers: {missing}")

    def _detect_harmful_stereotypes(self, text: str) -> list[str]:
        lowered = text.lower()

        patterns: dict[str, str] = {
            "blanket_group_generalization": r"\ball\s+(women|men|gay\s+people|trans\s+people|muslims|christians|jews|immigrants|black\s+people|asian\s+people|latinos)\s+(are|do|think)\b",
            "typical_group_generalization": r"\btypical\s+(female|male|gay|trans|muslim|christian|jewish|immigrant|black|asian|latino)\b",
            "you_people": r"\byou\s+people\b",
        }

        matches: list[str] = []
        for name, pattern in patterns.items():
            if re.search(pattern, lowered):
                matches.append(name)

        return matches

    def _crisis_response(self, base: str) -> str:
        return base.strip()

    def _cultural_response(self, base: str) -> str:
        return base.strip()

def main():
    """Test the edge case integrator"""
    integrator = EdgeCaseIntegrator()
    datasets = integrator.create_edge_case_datasets()
    print(f"Generated {len(datasets)} edge case integrated datasets")

if __name__ == "__main__":
    main()
