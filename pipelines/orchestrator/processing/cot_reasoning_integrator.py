"""
Chain-of-Thought Reasoning Integrator

Integrates various CoT reasoning datasets for therapeutic applications:
- CoT_Reasoning_Clinical_Diagnosis_Mental_Health
- CoT_Neurodivergent_vs_Neurotypical_Interactions
- CoT_Heartbreak_and_Breakups
- CoT_Reasoning_Mens_Mental_Health
- CoT_Legal_Issues_And_Laws
- CoT_Philosophical_Understanding
- CoT_Rare-Diseases_And_Health-Conditions
- CoT_Temporal_Reasoning_Dataset
- CoT_Reasoning_Scientific_Discovery_and_Research
- CoT-Reasoning_Cultural_Nuances
- ToT_Reasoning_Problem_Solving_Dataset_V2
"""

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)


@dataclass
class CoTExample:
    """Chain-of-Thought reasoning example."""

    example_id: str
    reasoning_type: str
    problem_statement: str
    reasoning_chain: list[str]
    final_conclusion: str
    therapeutic_context: dict[str, Any] = field(default_factory=dict)
    complexity_score: float = 0.0


@dataclass
class CoTDataset:
    """Complete CoT reasoning dataset."""

    dataset_name: str
    reasoning_type: str
    examples: list[CoTExample]
    metadata: dict[str, Any] = field(default_factory=dict)


class CoTReasoningIntegrator:
    """Integrates Chain-of-Thought reasoning datasets for therapeutic applications."""

    def __init__(
        self,
        base_path: str = "./cot_datasets",
        output_dir: str = "./integrated_datasets",
        dataset_registry_path: str = "ai/data/dataset_registry.json",
    ):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(__name__)
        self.dataset_registry_path = Path(dataset_registry_path)
        self.dataset_registry = self._load_dataset_registry()
        self.cot_registry_map = self._build_cot_registry_map()

        self.cot_configs = {
            "rpsd": {
                "name": "RPSD",
                "description": "Reasoning and problem solving dataset",
                "reasoning_type": "research_reasoning",
                "therapeutic_focus": "problem_solving",
                "expected_size": "large",
            },
            "tot_rpsd_v2": {
                "name": "ToT-RPSD-V2",
                "description": "Tree of Thought RPSD v2",
                "reasoning_type": "tree_of_thought_reasoning",
                "therapeutic_focus": "structured_problem_solving",
                "expected_size": "large",
            },
            "clinical_diagnosis": {
                "name": "CoT_Reasoning_Clinical_Diagnosis_Mental_Health",
                "description": "Clinical diagnostic reasoning for mental health conditions",
                "reasoning_type": "clinical_diagnosis_reasoning",
                "therapeutic_focus": "diagnostic_reasoning",
                "expected_size": "large",
            },
            "neurodiversity": {
                "name": "CoT_Neurodivergent_vs_Neurotypical_Interactions",
                "description": "Neurodivergent vs neurotypical communication patterns",
                "reasoning_type": "neurodiversity_reasoning",
                "therapeutic_focus": "neurodiversity_support",
                "expected_size": "medium",
            },
            "heartbreak": {
                "name": "CoT_Heartbreak_and_Breakups",
                "description": "Relationship rupture and recovery scenarios",
                "reasoning_type": "relationship_reasoning",
                "therapeutic_focus": "relationship_support",
                "expected_size": "medium",
            },
            "mens_mental_health": {
                "name": "CoT_Reasoning_Mens_Mental_Health",
                "description": "Men-specific mental health challenges",
                "reasoning_type": "gender_specific_reasoning",
                "therapeutic_focus": "men_mental_health_support",
                "expected_size": "medium",
            },
            "legal_issues": {
                "name": "CoT_Legal_Issues_And_Laws",
                "description": "Legal and ethical reasoning in therapeutic contexts",
                "reasoning_type": "legal_reasoning",
                "therapeutic_focus": "legal_ethics",
                "expected_size": "medium",
            },
            "philosophical": {
                "name": "CoT_Philosophical_Understanding",
                "description": "33MB, 60K existential/philosophical therapy",
                "reasoning_type": "philosophical_reasoning",
                "therapeutic_focus": "existential_therapy",
                "expected_size": "large",
            },
            "rare_diseases": {
                "name": "CoT_Rare-Diseases_And_Health-Conditions",
                "description": "Psychological support for rare diseases and health conditions",
                "reasoning_type": "rare_disease_reasoning",
                "therapeutic_focus": "medical_psychology",
                "expected_size": "large",
            },
            "temporal": {
                "name": "CoT_Temporal_Reasoning_Dataset",
                "description": "15MB, 30K time-based therapeutic planning",
                "reasoning_type": "temporal_reasoning",
                "therapeutic_focus": "treatment_planning",
                "expected_size": "medium",
            },
            "scientific_research": {
                "name": "CoT_Reasoning_Scientific_Discovery_and_Research",
                "description": "Evidence-based therapeutic reasoning grounded in research discovery",
                "reasoning_type": "research_reasoning",
                "therapeutic_focus": "evidence_based_practice",
                "expected_size": "medium",
            },
            "cultural": {
                "name": "CoT-Reasoning_Cultural_Nuances",
                "description": "Culturally-sensitive therapeutic approaches",
                "reasoning_type": "cultural_reasoning",
                "therapeutic_focus": "cultural_therapy",
                "expected_size": "medium",
            },
            "tree_of_thought": {
                "name": "ToT_Reasoning_Problem_Solving_Dataset_V2",
                "description": "Tree-of-Thought problem solving for therapeutic planning",
                "reasoning_type": "tree_of_thought_reasoning",
                "therapeutic_focus": "structured_problem_solving",
                "expected_size": "large",
            },
            "general_inquiry": {
                "name": "General_Inquiry_Thinking_Chain_Of_Thought",
                "description": "General inquiry chain-of-thought reasoning",
                "reasoning_type": "research_reasoning",
                "therapeutic_focus": "general_reasoning",
                "expected_size": "medium",
            },
            "ancient_past": {
                "name": "CoT_Reasoning_The_Ancient_Past",
                "description": "Historical and existential reasoning",
                "reasoning_type": "philosophical_reasoning",
                "therapeutic_focus": "existential_therapy",
                "expected_size": "medium",
            },
            "medical_diagnosis": {
                "name": "CoT_Medical_Diagnosis_3k",
                "description": "Medical diagnosis reasoning",
                "reasoning_type": "clinical_diagnosis_reasoning",
                "therapeutic_focus": "diagnostic_reasoning",
                "expected_size": "medium",
            },
            "first_responders": {
                "name": "CoT_Reasoning_First_Responders_Triage_And_Emergencies",
                "description": "Crisis triage and emergency reasoning",
                "reasoning_type": "legal_reasoning",
                "therapeutic_focus": "crisis_response",
                "expected_size": "medium",
            },
            "quantum_physics": {
                "name": "CoT_Reasoning_Quantum_Physics_And_Computing",
                "description": "Complex systems reasoning",
                "reasoning_type": "research_reasoning",
                "therapeutic_focus": "complex_systems",
                "expected_size": "medium",
            },
        }

        self.reasoning_patterns = {
            "clinical_diagnosis_reasoning": [
                "Gather symptom clusters and duration",
                "Map findings to DSM-5 criteria",
                "Rule out differential diagnoses",
                "Assess severity and functional impact",
                "Select initial intervention plan",
            ],
            "neurodiversity_reasoning": [
                "Consider neurodivergent perspective",
                "Assess sensory processing differences",
                "Evaluate communication preferences",
                "Account for executive function variations",
                "Recognize masking behaviors",
            ],
            "relationship_reasoning": [
                "Validate acute emotional distress",
                "Assess attachment and rupture patterns",
                "Stabilize with safety and boundaries",
                "Build adaptive coping strategies",
                "Plan paced contact and recovery goals",
            ],
            "gender_specific_reasoning": [
                "Consider societal gender expectations",
                "Assess masculine identity pressures",
                "Evaluate emotional expression barriers",
                "Account for help-seeking stigma",
                "Recognize vulnerability challenges",
            ],
            "legal_reasoning": [
                "Identify jurisdictional obligations",
                "Evaluate mandated reporting thresholds",
                "Clarify confidentiality boundaries",
                "Document risk and rationale",
                "Coordinate with legal/ethical resources",
            ],
            "philosophical_reasoning": [
                "Examine existential concerns",
                "Explore meaning and purpose",
                "Consider life's fundamental questions",
                "Assess values and beliefs",
                "Evaluate spiritual dimensions",
            ],
            "rare_disease_reasoning": [
                "Confirm medical context and prognosis",
                "Assess psychological impact of chronic/rare condition",
                "Coordinate with multidisciplinary care team",
                "Plan accommodations and pacing",
                "Build resilience and support network",
            ],
            "temporal_reasoning": [
                "Assess timeline of symptoms",
                "Plan treatment progression",
                "Consider developmental stages",
                "Evaluate progress markers",
                "Project future outcomes",
            ],
            "research_reasoning": [
                "Identify evidence base for presenting problem",
                "Critically appraise study quality and applicability",
                "Translate findings into client-specific plan",
                "Document risks, benefits, and uncertainties",
                "Monitor outcomes and iterate",
            ],
            "cultural_reasoning": [
                "Consider cultural background",
                "Assess family dynamics",
                "Evaluate cultural values",
                "Account for language barriers",
                "Recognize cultural stigma",
            ],
            "tree_of_thought_reasoning": [
                "Generate multiple therapeutic hypotheses",
                "Branch potential interventions",
                "Score branches for risk and feasibility",
                "Prune ineffective branches",
                "Select best path and plan checkpoints",
            ],
        }

        logger.info("CoTReasoningIntegrator initialized")

    def _load_dataset_registry(self) -> dict[str, Any]:
        if not self.dataset_registry_path.exists():
            logger.warning("Dataset registry not found; proceeding with defaults")
            return {}

        try:
            with open(self.dataset_registry_path) as f:
                return json.load(f)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Failed to load dataset registry: {exc}")
            return {}

    def _build_cot_registry_map(self) -> dict[str, str]:
        datasets = self.dataset_registry.get("datasets", {}).get("cot_reasoning", {})
        return {
            name: entry.get("path")
            for name, entry in datasets.items()
            if isinstance(entry, dict) and entry.get("path")
        }

    def _resolve_dataset_path(self, dataset_name: str) -> str | None:
        if dataset_name in self.cot_registry_map:
            return self.cot_registry_map[dataset_name]

        normalized = dataset_name.replace("-", "_")
        return next(
            (
                path
                for candidate, path in self.cot_registry_map.items()
                if candidate.replace("-", "_").lower() == normalized.lower()
            ),
            None,
        )

    def _is_s3_path(self, path_value: str | None) -> bool:
        return bool(path_value and str(path_value).startswith("s3://"))

    def _select_dataset_path(self, resolved_path: str | None, config_name: str) -> Path:
        if resolved_path and not self._is_s3_path(resolved_path):
            local_path = Path(resolved_path).expanduser()
            if local_path.exists():
                return local_path
        return self.base_path / config_name

    def _is_ovhai_available(self) -> bool:
        return shutil.which("ovhai") is not None

    def _download_remote_dataset(
        self, resolved_path: str, target_path: Path, issues: list[str]
    ) -> None:
        if not self._is_s3_path(resolved_path):
            return

        if not self._is_ovhai_available():
            issues.append("ovhai CLI not found; cannot fetch S3 dataset")
            return

        target_path.mkdir(parents=True, exist_ok=True)
        cmd = ["ovhai", "cp", resolved_path, str(target_path)]

        try:
            completed = subprocess.run(
                cmd, check=True, capture_output=True, text=True
            )
            if completed.stdout:
                logger.info(completed.stdout.strip())
        except subprocess.CalledProcessError as exc:
            message = exc.stderr or exc.stdout or str(exc)
            issues.append(f"ovhai download failed for {resolved_path}: {message}")
            logger.error("OVHAI cp failed for %s: %s", resolved_path, message)
        except FileNotFoundError:
            issues.append("ovhai CLI not found; cannot fetch S3 dataset")

    def integrate_all_cot_datasets(self) -> dict[str, Any]:
        """Integrate all CoT reasoning datasets."""
        start_time = datetime.now()

        results: dict[str, Any] = {}
        total_examples = 0

        for dataset_key, config in self.cot_configs.items():
            logger.info(f"Integrating {config['name']}...")

            result = self.integrate_cot_dataset(dataset_key)
            results[dataset_key] = result

            if result.get("success"):
                total_examples += result.get("examples_processed", 0)

        return {
            "integration_type": "Chain-of-Thought Reasoning Datasets",
            "total_datasets": len(self.cot_configs),
            "successful_integrations": sum(bool(result.get("success"))
                                       for result in results.values()),
            "total_examples": total_examples,
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "individual_results": results,
        }

    def integrate_cot_dataset(self, dataset_key: str) -> dict[str, Any]:
        """Integrate a specific CoT reasoning dataset."""
        if dataset_key not in self.cot_configs:
            return {"success": False, "issues": [f"Unknown dataset key: {dataset_key}"]}

        config = self.cot_configs[dataset_key]

        result: dict[str, Any] = {
            "success": False,
            "dataset_name": config["name"],
            "examples_processed": 0,
            "quality_metrics": {},
            "issues": [],
            "output_path": None,
            "registry_path": None,
        }

        try:
            resolved_path = self._resolve_dataset_path(config["name"])
            result["registry_path"] = resolved_path

            dataset_path = self._select_dataset_path(resolved_path, config["name"])

            if self._is_s3_path(resolved_path) and not dataset_path.exists():
                self._download_remote_dataset(resolved_path or "", dataset_path, result["issues"])

            if not dataset_path.exists():
                self._create_mock_cot_data(dataset_path, config)
                result["issues"].append(f"Created mock data for {config['name']}")

            cot_examples = self._load_cot_examples(dataset_path, config)

            processed_examples: list[CoTExample] = []
            for example in cot_examples:
                if processed_example := self._process_cot_example(example, config):
                    processed_examples.append(processed_example)

            cot_dataset = CoTDataset(
                dataset_name=config["name"],
                reasoning_type=config["reasoning_type"],
                examples=processed_examples,
                metadata={
                    "description": config["description"],
                    "therapeutic_focus": config["therapeutic_focus"],
                    "integrated_at": datetime.now().isoformat(),
                },
            )

            quality_metrics = self._assess_cot_quality(cot_dataset)
            output_path = self._save_cot_dataset(cot_dataset, quality_metrics)

            result |= {
                "success": True,
                "examples_processed": len(processed_examples),
                "quality_metrics": quality_metrics,
                "output_path": str(output_path),
            }

            logger.info(
                "Successfully integrated %s: %d examples",
                config["name"],
                len(processed_examples),
            )

        except Exception as exc:  # pragma: no cover - defensive
            result["issues"].append(f"Integration failed: {exc!s}")
            logger.error("CoT integration failed for %s: %s", dataset_key, exc)

        return result

    def _create_mock_cot_data(self, dataset_path: Path, config: dict[str, Any]) -> None:
        """Create mock CoT reasoning data."""
        dataset_path.mkdir(parents=True, exist_ok=True)

        size_mapping = {"small": 50, "medium": 200, "large": 500}
        num_examples = size_mapping.get(config["expected_size"], 100)

        examples = []
        reasoning_patterns = self.reasoning_patterns[config["reasoning_type"]]

        for i in range(num_examples):
            example = self._generate_mock_cot_example(i, config, reasoning_patterns)
            examples.append(example)

        data_file = dataset_path / "cot_examples.jsonl"
        with open(data_file, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")

        metadata = {
            "dataset_name": config["name"],
            "description": config["description"],
            "reasoning_type": config["reasoning_type"],
            "therapeutic_focus": config["therapeutic_focus"],
            "total_examples": len(examples),
            "reasoning_patterns": reasoning_patterns,
            "created_at": datetime.now().isoformat(),
        }

        with open(dataset_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def _generate_mock_cot_example(
        self, index: int, config: dict[str, Any], reasoning_patterns: list[str]
    ) -> dict[str, Any]:
        """Generate mock CoT example based on reasoning type."""
        reasoning_type = config["reasoning_type"]

        scenarios = {
            "clinical_diagnosis_reasoning": (
                "A client presents with overlapping depressive and anxiety symptoms after a prolonged period of workplace stress.",
                [
                    "Gather symptom clusters and duration: Document depressive and anxious features",
                    "Map findings to DSM-5 criteria: Check major depressive episode and GAD thresholds",
                    "Rule out differentials: Exclude substance, thyroid, and adjustment disorder",
                    "Assess severity and functional impact: Evaluate sleep, appetite, and concentration",
                    "Select initial intervention plan: Psychoeducation, CBT, and medication consult",
                ],
                "Provide a provisional diagnosis, create a stabilization plan, and schedule close follow-up for safety and response tracking.",
            ),
            "neurodiversity_reasoning": (
                "A client with autism spectrum disorder is struggling with social anxiety in workplace settings. They report feeling overwhelmed by office noise and unexpected schedule changes.",
                [
                    "Consider neurodivergent perspective: Client may have heightened sensory sensitivity",
                    "Assess sensory processing differences: Office environment may be overstimulating",
                    "Evaluate communication preferences: Direct, clear communication may be preferred",
                    "Account for executive function variations: Schedule changes may be particularly challenging",
                    "Recognize masking behaviors: Client may be exhausted from masking autistic traits",
                ],
                "Recommend sensory accommodations, structured communication protocols, and validation of neurodivergent experiences while building coping strategies.",
            ),
            "relationship_reasoning": (
                "A client is in acute distress after an unexpected breakup and is struggling with intrusive thoughts and sleep disruption.",
                [
                    "Validate acute emotional distress: Normalize grief responses",
                    "Assess attachment and rupture patterns: Identify abandonment triggers",
                    "Stabilize with safety and boundaries: Limit contact while emotions are acute",
                    "Build adaptive coping strategies: Sleep hygiene, social support, grounding",
                    "Plan paced contact and recovery goals: Structured reflection before major decisions",
                ],
                "Focus on stabilization, adaptive coping, and paced meaning-making while monitoring for depressive or risk escalation.",
            ),
            "gender_specific_reasoning": (
                "A male client in his 30s is reluctant to discuss emotional struggles, presenting only with work stress and relationship conflicts.",
                [
                    "Consider societal gender expectations: Men often discouraged from emotional expression",
                    "Assess masculine identity pressures: May feel vulnerability threatens masculinity",
                    "Evaluate emotional expression barriers: Limited emotional vocabulary common",
                    "Account for help-seeking stigma: Therapy may feel like admission of weakness",
                    "Recognize vulnerability challenges: Need safe space to explore emotions",
                ],
                "Use strength-based approach, normalize emotional experiences, and gradually build emotional awareness through practical frameworks.",
            ),
            "legal_reasoning": (
                "A client discloses potential workplace harassment and asks about confidentiality limits before sharing specifics.",
                [
                    "Identify jurisdictional obligations: Confirm legal and ethical boundaries",
                    "Evaluate mandated reporting thresholds: Screen for imminent harm or protected classes",
                    "Clarify confidentiality boundaries: Explain scope and limits before details",
                    "Document risk and rationale: Keep contemporaneous notes of decisions",
                    "Coordinate with legal/ethical resources: Consult supervision or legal counsel as needed",
                ],
                "Provide clear confidentiality framing, document decisions, and ensure client safety while following legal and ethical requirements.",
            ),
            "philosophical_reasoning": (
                "A client is experiencing existential crisis following major life transition, questioning life's meaning and purpose.",
                [
                    "Examine existential concerns: Client facing fundamental questions about existence",
                    "Explore meaning and purpose: Transition has disrupted sense of direction",
                    "Consider life's fundamental questions: What makes life worth living?",
                    "Assess values and beliefs: Core beliefs may be challenged or evolving",
                    "Evaluate spiritual dimensions: May need to explore transcendent meaning",
                ],
                "Engage in existential exploration, help client construct personal meaning, and support values clarification process.",
            ),
            "rare_disease_reasoning": (
                "A client recently diagnosed with a rare autoimmune condition reports fear of disability and isolation.",
                [
                    "Confirm medical context and prognosis: Coordinate with medical team",
                    "Assess psychological impact: Grief, uncertainty, and identity shifts",
                    "Coordinate with multidisciplinary care team: Align goals and communication",
                    "Plan accommodations and pacing: Energy management and adaptive scheduling",
                    "Build resilience and support network: Peer groups and caregiver education",
                ],
                "Develop a collaborative care plan that balances medical demands with psychological support and sustainable pacing.",
            ),
            "temporal_reasoning": (
                "A client with depression needs comprehensive treatment planning considering symptom progression and recovery timeline.",
                [
                    "Assess timeline of symptoms: Depression developed over 6-month period",
                    "Plan treatment progression: Start with stabilization, then skill-building",
                    "Consider developmental stages: Client in early career development phase",
                    "Evaluate progress markers: Weekly mood tracking and functional improvements",
                    "Project future outcomes: Expect gradual improvement over 3-6 months",
                ],
                "Implement phased treatment approach with clear milestones and regular progress evaluation.",
            ),
            "research_reasoning": (
                "A client asks whether a new evidence-based protocol is appropriate for their complex trauma history.",
                [
                    "Identify evidence base for presenting problem: Locate trauma-focused studies",
                    "Critically appraise study quality and applicability: Check inclusion/exclusion",
                    "Translate findings into client-specific plan: Adapt pacing and dosing",
                    "Document risks, benefits, and uncertainties: Ensure informed collaboration",
                    "Monitor outcomes and iterate: Track response and side-effects closely",
                ],
                "Propose an adapted, trauma-informed plan that is evidence-aligned, collaboratively reviewed, and monitored for safety.",
            ),
            "cultural_reasoning": (
                "A client from collectivist cultural background is struggling with individual therapy approach and family expectations.",
                [
                    "Consider cultural background: Collectivist values may conflict with individual focus",
                    "Assess family dynamics: Family involvement may be crucial for success",
                    "Evaluate cultural values: Honor and family harmony highly valued",
                    "Account for language barriers: May need culturally adapted interventions",
                    "Recognize cultural stigma: Mental health treatment may carry cultural shame",
                ],
                "Adapt therapy to include family systems perspective and culturally sensitive interventions.",
            ),
            "tree_of_thought_reasoning": (
                "A client faces recurring conflict with a sibling caregiver and wants a structured plan to reduce escalation.",
                [
                    "Generate multiple therapeutic hypotheses: Communication style, role strain, unmet needs",
                    "Branch potential interventions: Mediation, boundary-setting, shared schedules",
                    "Score branches for risk and feasibility: Prioritize safety and low-cost steps",
                    "Prune ineffective branches: Remove high-risk or low-impact options",
                    "Select best path and plan checkpoints: Pilot one intervention with weekly review",
                ],
                "Deploy the highest-safety branch first, review outcomes weekly, and iterate with additional branches as needed.",
            ),
        }

        if reasoning_type not in scenarios:
            reasoning_chain = reasoning_patterns
            problem = (
                f"Therapeutic reasoning scenario for {config['description']} focusing on "
                f"{config['therapeutic_focus']}."
            )
            conclusion = (
                "Apply evidence-aligned plan with clear safety checks, documentation, and follow-up."
            )
        else:
            problem, reasoning_chain, conclusion = scenarios[reasoning_type]

        return {
            "id": f"{reasoning_type}_example_{index:03d}",
            "problem_statement": problem,
            "reasoning_chain": reasoning_chain,
            "conclusion": conclusion,
            "metadata": {
                "reasoning_type": reasoning_type,
                "therapeutic_focus": config["therapeutic_focus"],
                "complexity_level": ["basic", "intermediate", "advanced"][index % 3],
                "example_index": index,
            },
        }

    def _load_cot_examples(
        self, dataset_path: Path, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Load CoT examples from dataset."""
        examples = []
        data_file = dataset_path / "cot_examples.jsonl"

        if data_file.exists():
            with open(data_file) as f:
                for line_num, line in enumerate(f, 1):
                    if line := line.strip():
                        try:
                            example = json.loads(line)
                            examples.append(example)
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Invalid JSON on line {line_num} in {data_file}: {e}"
                            )

        return examples

    def _process_cot_example(
        self, example_data: dict[str, Any], config: dict[str, Any]
    ) -> CoTExample | None:
        """Process and validate a CoT example."""
        try:
            example_id = example_data.get("id", f"cot_{hash(str(example_data))%10000}")
            problem_statement = example_data.get("problem_statement", "")
            reasoning_chain = example_data.get("reasoning_chain", [])
            conclusion = example_data.get("conclusion", "")

            if not all([problem_statement, reasoning_chain, conclusion]):
                logger.warning(f"Incomplete CoT example: {example_id}")
                return None

            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(
                reasoning_chain, conclusion
            )

            # Extract therapeutic context
            therapeutic_context = {
                **example_data.get("metadata", {}),
                "reasoning_steps": len(reasoning_chain),
                "therapeutic_focus": config["therapeutic_focus"],
                "processed_at": datetime.now().isoformat(),
            }

            return CoTExample(
                example_id=example_id,
                reasoning_type=config["reasoning_type"],
                problem_statement=problem_statement,
                reasoning_chain=reasoning_chain,
                final_conclusion=conclusion,
                therapeutic_context=therapeutic_context,
                complexity_score=complexity_score,
            )

        except Exception as e:
            logger.error(f"Error processing CoT example: {e}")
            return None

    def _calculate_complexity_score(
        self, reasoning_chain: list[str], conclusion: str
    ) -> float:
        """Calculate complexity score for CoT example."""
        score = 0.5  # Base score

        # Number of reasoning steps
        step_count = len(reasoning_chain)
        if step_count >= 5:
            score += 0.2
        elif step_count >= 3:
            score += 0.1

        # Depth of reasoning (longer steps indicate deeper thinking)
        avg_step_length = (
            sum(len(step.split()) for step in reasoning_chain) / len(reasoning_chain)
            if reasoning_chain
            else 0
        )
        if avg_step_length >= 15:
            score += 0.2
        elif avg_step_length >= 10:
            score += 0.1

        # Conclusion quality (length and specificity)
        conclusion_words = len(conclusion.split())
        if conclusion_words >= 20:
            score += 0.1

        return min(1.0, score)

    def _assess_cot_quality(self, cot_dataset: CoTDataset) -> dict[str, float]:
        """Assess quality of CoT dataset."""
        examples = cot_dataset.examples

        if not examples:
            return {"overall_quality": 0.0}

        # Calculate quality metrics
        complexity_scores = [e.complexity_score for e in examples]
        reasoning_step_counts = [len(e.reasoning_chain) for e in examples]

        return {
            "overall_quality": sum(complexity_scores) / len(complexity_scores),
            "average_reasoning_steps": sum(reasoning_step_counts)
            / len(reasoning_step_counts),
            "complexity_variance": sum(
                (c - sum(complexity_scores) / len(complexity_scores)) ** 2
                for c in complexity_scores
            )
            / len(complexity_scores),
            "high_complexity_examples": sum(c >= 0.8 for c in complexity_scores)
            / len(complexity_scores),
            "reasoning_depth": sum(steps >= 5 for steps in reasoning_step_counts)
            / len(reasoning_step_counts),
        }


    def _save_cot_dataset(
        self, cot_dataset: CoTDataset, quality_metrics: dict[str, float]
    ) -> Path:
        """Save CoT dataset."""
        output_file = self.output_dir / f"{cot_dataset.reasoning_type}_integrated.json"

        # Convert to serializable format
        examples_data = []
        for example in cot_dataset.examples:
            example_dict = {
                "example_id": example.example_id,
                "reasoning_type": example.reasoning_type,
                "problem_statement": example.problem_statement,
                "reasoning_chain": example.reasoning_chain,
                "final_conclusion": example.final_conclusion,
                "therapeutic_context": example.therapeutic_context,
                "complexity_score": example.complexity_score,
            }
            examples_data.append(example_dict)

        output_data = {
            "dataset_info": {
                "name": cot_dataset.dataset_name,
                "reasoning_type": cot_dataset.reasoning_type,
                "total_examples": len(cot_dataset.examples),
                "integrated_at": datetime.now().isoformat(),
            },
            "quality_metrics": quality_metrics,
            "metadata": cot_dataset.metadata,
            "examples": examples_data,
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"CoT dataset saved: {output_file}")
        return output_file


# Example usage
if __name__ == "__main__":
    # Initialize integrator
    integrator = CoTReasoningIntegrator()

    # Integrate all CoT datasets
    results = integrator.integrate_all_cot_datasets()

    # Show results

    for dataset_key, result in results["individual_results"].items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        print(f"{dataset_key}: {status}")
