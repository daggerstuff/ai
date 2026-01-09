#!/usr/bin/env python3
"""
Edge Case Integrator - KAN-28 Component #3
Integrates nightmare fuel scenarios with expert voices and therapeutic frameworks
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

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

    def __init__(self, edge_case_dirs: Optional[Iterable[str]] = None) -> None:
        default_dirs = [
            "ai/training_data_consolidated/edge_cases/",
            "ai/training_data_consolidated/edge_cases_cultural/",
            "ai/training_data_consolidated/crisiscultural/",
        ]
        self.edge_case_dirs: List[Path] = [
            Path(p) for p in (edge_case_dirs or default_dirs)
        ]
        self.scenarios: List[Dict[str, Any]] = []

    def load_existing_edge_cases(self) -> List[Dict[str, Any]]:
        """Load existing edge case scenarios from configured directories.

        Returns a list of scenario dicts. If nothing is found, returns sample cases.
        """

        edge_cases: List[Dict[str, Any]] = []

        any_dir_exists = False
        for directory in self.edge_case_dirs:
            if not directory.exists():
                logger.info("Edge case directory not found (skipping): %s", directory)
                continue
            any_dir_exists = True
            for file_path in directory.glob("*.json*"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        if file_path.suffix == ".jsonl":
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                try:
                                    parsed = json.loads(line)
                                    if isinstance(parsed, list):
                                        edge_cases.extend(parsed)
                                    else:
                                        edge_cases.append(parsed)
                                except Exception as line_err:
                                    logger.warning(
                                        "Skipping malformed JSONL line in %s: %s",
                                        file_path,
                                        line_err,
                                    )
                        else:
                            loaded = json.load(f)
                            if isinstance(loaded, list):
                                edge_cases.extend(loaded)
                            elif isinstance(loaded, dict):
                                edge_cases.append(loaded)
                            else:
                                logger.warning(
                                    "Unsupported JSON root type in %s: %s",
                                    file_path,
                                    type(loaded),
                                )
                except Exception as e:
                    logger.warning("Could not load edge case file %s: %s", file_path, e)

        if not any_dir_exists or not edge_cases:
            logger.warning(
                "No edge case files found in configured directories. "
                "Using sample scenarios."
            )
            return self._create_sample_edge_cases()

        return edge_cases

    def load_keyword_resources(
        self, dirs: Optional[Iterable[str]] = None
    ) -> Dict[str, List[str]]:
        """Load cultural crisis keyword resources from JSON/JSONL files.

        Supports flexible schemas:
        - Object with keys like 'suicide_euphemisms' (or 'euphemisms', 'phrases')
          and 'safe_idiomatic_exclusions' (or 'exclusions', 'patterns', 'regex').
        - List of strings: treated as euphemisms.
        - JSONL with one object per line containing 'type' and fields like 'text',
          'phrase' (for euphemisms) or 'pattern'/'regex' (for exclusions).
        """
        sources = [Path(p) for p in dirs] if dirs else self.edge_case_dirs
        euphemisms: Set[str] = set()
        exclusions: Set[str] = set()

        def _maybe_add_euph(value: Any) -> None:
            if isinstance(value, str):
                s = value.strip()
                if s:
                    euphemisms.add(s)
            elif isinstance(value, list):
                for v in value:
                    _maybe_add_euph(v)

        def _maybe_add_excl(value: Any) -> None:
            if isinstance(value, str):
                s = value.strip()
                if not s:
                    return
                # Validate regex compiles in Python (best-effort); skip if invalid
                try:
                    re.compile(s)
                    exclusions.add(s)
                except re.error:
                    logger.warning("Skipping invalid regex pattern: %s", s)
            elif isinstance(value, list):
                for v in value:
                    _maybe_add_excl(v)

        for directory in sources:
            if not directory.exists():
                continue
            for file_path in directory.glob("*.json*"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        if file_path.suffix == ".jsonl":
                            for raw in f:
                                raw = raw.strip()
                                if not raw:
                                    continue
                                try:
                                    obj = json.loads(raw)
                                except Exception as err:
                                    logger.warning(
                                        "Skipping malformed JSONL in %s: %s",
                                        file_path,
                                        err,
                                    )
                                    continue
                                if isinstance(obj, dict):
                                    t = str(obj.get("type", "")).lower()
                                    if t in {
                                        "euphemism",
                                        "suicide_euphemism",
                                        "phrase",
                                    }:
                                        _maybe_add_euph(
                                            obj.get("text")
                                            or obj.get("phrase")
                                            or obj.get("value")
                                        )
                                    elif t in {
                                        "exclusion",
                                        "idiom",
                                        "regex",
                                        "pattern",
                                    }:
                                        _maybe_add_excl(
                                            obj.get("pattern")
                                            or obj.get("regex")
                                            or obj.get("value")
                                        )
                                    else:
                                        # Heuristic fields
                                        if (
                                            "suicide_euphemisms" in obj
                                            or "euphemisms" in obj
                                        ):
                                            _maybe_add_euph(
                                                obj.get("suicide_euphemisms")
                                                or obj.get("euphemisms")
                                            )
                                        if (
                                            "safe_idiomatic_exclusions" in obj
                                            or "exclusions" in obj
                                            or "patterns" in obj
                                            or "regex" in obj
                                        ):
                                            _maybe_add_excl(
                                                obj.get("safe_idiomatic_exclusions")
                                                or obj.get("exclusions")
                                                or obj.get("patterns")
                                                or obj.get("regex")
                                            )
                                elif isinstance(obj, list):
                                    # Assume euphemisms if flat list of strings
                                    _maybe_add_euph(obj)
                        else:
                            data = json.load(f)
                            if isinstance(data, dict):
                                _maybe_add_euph(
                                    data.get("suicide_euphemisms")
                                    or data.get("euphemisms")
                                    or data.get("phrases")
                                )
                                _maybe_add_excl(
                                    data.get("safe_idiomatic_exclusions")
                                    or data.get("exclusions")
                                    or data.get("patterns")
                                    or data.get("regex")
                                )
                            elif isinstance(data, list):
                                _maybe_add_euph(data)
                except Exception as e:
                    logger.warning("Failed reading %s: %s", file_path, e)

        return {
            "suicide_euphemisms": sorted(euphemisms),
            "safe_idiomatic_exclusions": sorted(exclusions),
        }

    def write_keywords_artifacts(
        self,
        keywords: Dict[str, List[str]],
        output_dir: str = "ai/training_data_consolidated/crisis_keywords/",
    ) -> Dict[str, Path]:
        """Write consolidated keyword artifacts as JSON and optional TS module.

        Returns a dict with paths to generated artifacts.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        json_path = out_dir / "generated_keywords.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(keywords, f, ensure_ascii=False, indent=2)

        # Optionally generate a TS module for easy import in the app
        ts_dir = Path("src/lib/ai/crisis/config")
        ts_dir.mkdir(parents=True, exist_ok=True)
        ts_path = ts_dir / "generated.keywords.ts"

        euphs = keywords.get("suicide_euphemisms", [])
        excl = keywords.get("safe_idiomatic_exclusions", [])

        # Create RegExp constructors with 'i' flag; escape backticks in patterns
        def _js_string(s: str) -> str:
            return s.replace("`", "\\`")

        ts_content = [
            "// AUTO-GENERATED by edge_case_integrator.py — do not edit manually",
            "export const SUICIDE_EUPHEMISMS = [",
        ]
        for item in euphs:
            ts_content.append(f"  '{_js_string(item)}',")
        ts_content.append("] as const\n")

        ts_content.append("export const SAFE_IDIOMATIC_EXCLUSIONS = [\n")
        for pat in excl:
            # Wrap raw pattern; JS flags: i
            ts_content.append(f"  new RegExp('{_js_string(pat)}', 'i'),")
        ts_content.append("] as const\n")

        with open(ts_path, "w", encoding="utf-8") as f:
            f.write("\n".join(ts_content))

        logger.info("Wrote keyword artifacts: %s (JSON), %s (TS)", json_path, ts_path)
        return {"json": json_path, "ts": ts_path}

    def create_keywords_dataset(
        self,
        output_dir: str = "ai/training_data_consolidated/crisis_keywords/",
        source_dirs: Optional[Iterable[str]] = None,
    ) -> Dict[str, List[str]]:
        """Create consolidated keywords dataset and write artifacts.

        Returns the consolidated keyword dict with lists.
        """
        keywords = self.load_keyword_resources(dirs=source_dirs)
        self.write_keywords_artifacts(keywords, output_dir=output_dir)
        logger.info(
            "Consolidated keywords — euphemisms: %d, exclusions: %d",
            len(keywords.get("suicide_euphemisms", [])),
            len(keywords.get("safe_idiomatic_exclusions", [])),
        )
        return keywords

    def _create_sample_edge_cases(self) -> List[Dict[str, Any]]:
        """Create sample edge case scenarios for testing"""

        return [
            {
                "scenario_type": "suicidal_ideation",
                "severity_level": 9,
                "client_presentation": (
                    "I've been thinking about ending my life. I have a plan and I "
                    "don't see any other way out."
                ),
                "safety_considerations": [
                    "immediate_risk_assessment",
                    "safety_planning",
                    "emergency_protocols",
                ],
                "therapeutic_goals": [
                    "establish_safety",
                    "build_hope",
                    "create_support_network",
                ],
            },
            {
                "scenario_type": "trauma_flashback",
                "severity_level": 8,
                "client_presentation": (
                    "I'm having flashbacks again. I can't tell what's real anymore. "
                    "I feel like I'm back there."
                ),
                "safety_considerations": [
                    "grounding_techniques",
                    "present_moment_awareness",
                    "safety_validation",
                ],
                "therapeutic_goals": [
                    "establish_present_safety",
                    "trauma_integration",
                    "nervous_system_regulation",
                ],
            },
            {
                "scenario_type": "severe_dissociation",
                "severity_level": 7,
                "client_presentation": (
                    "I feel like I'm watching myself from outside my body. Nothing "
                    "feels real. I can't connect to anything."
                ),
                "safety_considerations": [
                    "grounding_interventions",
                    "co_regulation",
                    "gentle_presence",
                ],
                "therapeutic_goals": [
                    "embodied_presence",
                    "nervous_system_safety",
                    "gradual_integration",
                ],
            },
            {
                "scenario_type": "relationship_crisis",
                "severity_level": 6,
                "client_presentation": (
                    "My partner left me and I feel completely worthless. I don't "
                    "know who I am without them."
                ),
                "safety_considerations": [
                    "attachment_wounds",
                    "identity_stability",
                    "support_systems",
                ],
                "therapeutic_goals": [
                    "identity_rebuilding",
                    "attachment_healing",
                    "self_worth_restoration",
                ],
            },
            {
                "scenario_type": "addiction_relapse",
                "severity_level": 8,
                "client_presentation": (
                    "I relapsed last night after 6 months sober. I feel like a "
                    "complete failure and want to give up."
                ),
                "safety_considerations": [
                    "relapse_prevention",
                    "shame_spirals",
                    "motivation_rebuilding",
                ],
                "therapeutic_goals": [
                    "shame_resilience",
                    "progress_reframing",
                    "recovery_recommitment",
                ],
            },
        ]

    def integrate_with_expert_voices(
        self, edge_cases: List[Dict], expert_voices: Dict
    ) -> List[Dict[str, Any]]:
        """Integrate edge cases with tri-expert therapeutic responses"""

        integrated_scenarios = []

        for case in edge_cases:
            # Generate expert responses for this edge case
            expert_responses = self._generate_expert_responses_for_case(
                case, expert_voices
            )

            # Create integrated scenario
            integrated = {
                **case,
                "expert_responses": expert_responses,
                "integrated_response": self._create_integrated_response(
                    case, expert_responses
                ),
                "safety_protocol": self._create_safety_protocol(case),
                "therapeutic_framework": self._create_therapeutic_framework(case),
            }

            integrated_scenarios.append(integrated)

        return integrated_scenarios

    def _generate_expert_responses_for_case(
        self, case: Dict, expert_voices: Dict
    ) -> Dict[str, str]:
        """Generate responses from each expert for the edge case"""

        scenario_type = case.get("scenario_type", "")

        responses = {}

        # Tim Ferriss approach - systematic, fear-setting, actionable
        if scenario_type == "suicidal_ideation":
            self._extracted_from__generate_expert_responses_for_case_12(
                "Right now, we need to create a systematic safety plan. What would "
                "need to happen in the next 24 hours for you to feel 1% safer? "
                "Let's design the minimum effective dose of support.",
                responses,
                "Your pain is real and valid. What happened to you that taught you "
                "that your life doesn't matter? Your body is trying to protect you "
                "- let's listen to what it needs.",
                "You are worthy of love and belonging, even in this dark moment. "
                "Shame grows in silence - thank you for having the courage to share "
                "this with me.",
            )
        elif scenario_type == "trauma_flashback":
            self._extracted_from__generate_expert_responses_for_case_12(
                "Your brain is trying to protect you with outdated information. "
                "What would grounding look like if it were easy? Let's create a "
                "simple system you can use.",
                responses,
                "Your nervous system is responding to old wounds. When did you "
                "first learn that the world wasn't safe? Let's help your body "
                "remember that you're here now, not back there.",
                "You're being incredibly brave by staying present with me right "
                "now. Vulnerability is not weakness - it's your pathway back to "
                "wholeness.",
            )
        else:
            self._extracted_from__generate_expert_responses_for_case_12(
                "What's the smallest step we could take right now that would move "
                "you 1% in the direction of safety? Let's make this systematic and "
                "achievable.",
                responses,
                "What happened to you? Your pain makes sense in the context of your "
                "story. Let's explore this with compassion for the part of you "
                "that's struggling.",
                "What story are you telling yourself about your worth right now? "
                "You belong here, and your struggle doesn't define your value.",
            )
        return responses

    # TODO Rename this here and in `_generate_expert_responses_for_case`
    def _extracted_from__generate_expert_responses_for_case_12(
        self, arg0, responses, arg2, arg3
    ):
        responses["tim"] = arg0
        responses["gabor"] = arg2
        responses["brene"] = arg3

    def _create_integrated_response(self, case: Dict, expert_responses: Dict) -> str:
        """Create a blended response using all three expert approaches"""

        scenario_type = case.get("scenario_type", "")

        if scenario_type == "suicidal_ideation":
            return (
                "Your pain is real and you matter. Let's create a safety plan "
                "together - what would the next 24 hours look like if we designed "
                "them for your wellbeing? You're being incredibly brave by sharing "
                "this. Your life has value, and we're going to take this one small "
                "step at a time."
            )

        elif scenario_type == "trauma_flashback":
            return (
                "Your nervous system is doing its job - trying to protect you. "
                "Let's help your body remember you're safe here with me now. What "
                "would grounding feel like if it were gentle and easy? You're "
                "showing incredible courage by staying present."
            )

        else:
            return (
                "What you're experiencing makes complete sense given your story. "
                "Let's approach this with both compassion and practical steps. You "
                "are worthy of support, and we're going to figure this out "
                "together, one small step at a time."
            )

    def _create_safety_protocol(self, case: Dict) -> Dict[str, Any]:
        """Create safety protocol for the edge case"""

        return {
            "immediate_actions": case.get("safety_considerations", []),
            "risk_level": case.get("severity_level", 5),
            "emergency_contacts": [
                "crisis_hotline",
                "emergency_services",
                "trusted_support",
            ],
            "follow_up_required": case.get("severity_level", 5) >= 7,
        }

    def _create_therapeutic_framework(self, case: Dict) -> Dict[str, Any]:
        """Create therapeutic framework for the edge case"""

        return {
            "primary_goals": case.get("therapeutic_goals", []),
            "therapeutic_modalities": [
                "trauma_informed",
                "attachment_based",
                "somatic_experiencing",
            ],
            "session_structure": "safety_first_then_processing",
            "progress_markers": [
                "safety_increase",
                "symptom_reduction",
                "functional_improvement",
            ],
        }

    def create_edge_case_datasets(
        self, output_path: str = "ai/training_data_consolidated/edge_cases_enhanced/"
    ) -> List[Dict[str, Any]]:
        """Create integrated edge case datasets"""

        # Load existing edge cases
        edge_cases = self.load_existing_edge_cases()

        # For now, use simplified expert voices structure
        expert_voices = {
            "tim": {"style": "systematic_actionable"},
            "gabor": {"style": "trauma_informed_compassionate"},
            "brene": {"style": "shame_resilient_vulnerable"},
        }

        # Integrate with expert voices
        integrated_datasets = self.integrate_with_expert_voices(
            edge_cases, expert_voices
        )

        # Save datasets
        Path(output_path).mkdir(parents=True, exist_ok=True)
        output_file = Path(output_path) / "edge_cases_integrated.jsonl"

        with open(output_file, "w") as f:
            for dataset in integrated_datasets:
                f.write(json.dumps(dataset) + "\n")

        logger.info(
            "Created %s edge case integrated datasets at %s",
            len(integrated_datasets),
            output_file,
        )
        return integrated_datasets


def main():
    """Test the edge case integrator and generate keyword artifacts."""
    integrator = EdgeCaseIntegrator()
    datasets = integrator.create_edge_case_datasets()
    print(f"Generated {len(datasets)} edge case integrated datasets")

    # Create consolidated cultural keyword artifacts (JSON + TS)
    keywords = integrator.create_keywords_dataset()
    print(
        "Consolidated keywords — euphemisms: {e}, exclusions: {x}".format(
            e=len(keywords.get("suicide_euphemisms", [])),
            x=len(keywords.get("safe_idiomatic_exclusions", [])),
        )
    )


if __name__ == "__main__":
    main()
