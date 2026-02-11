import json
import logging
import random
from pathlib import Path

from .formatters import to_alpaca, to_sharegpt

logger = logging.getLogger(__name__)


class DatasetSynthesizer:
    """
    Synthesizes diverse raw and reasoning datasets into instruction tuning formats.
    Task 4.1 in Mental Health Datasets Expansion.
    """

    def __init__(self, base_path: str = "ai/training/ready_packages/datasets"):
        self.base_path = Path(base_path)
        self.stage2_path = self.base_path / "stage2_reasoning"
        self.stage4_path = self.base_path / "stage4_cleaning"
        self.output_path = self.base_path / "final_instruct"
        self._ensure_directories()

        # System prompts
        self.DEFAULT_SYSTEM_PROMPT = (
            "You are a specialized mental health AI assistant. "
            "Your goal is to provide empathetic, clinically grounded, and safe support."
        )

    def _ensure_directories(self):
        self.output_path.mkdir(parents=True, exist_ok=True)

    def load_json_files(self, directory: Path) -> list[dict]:
        """Recursively loads all JSON files from a directory."""
        data = []
        if not directory.exists():
            logger.warning(f"Directory not found: {directory}")
            return data

        for file_path in directory.rglob("*.json"):
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        data.extend(content)
                    else:
                        data.append(content)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        return data

    def synthesize_dataset(self, format_type: str = "alpaca") -> list[dict]:
        """
        Loads all data, transforms it, and returns a list of formatted items.
        """
        all_items = []

        # 1. Load Therapies (CBT, DBT, EMDR, ACT)
        logger.info("Loading Therapeutic Modality data...")
        therapies_data = self.load_json_files(self.stage2_path)

        # 2. Transform Therapies
        for item in therapies_data:
            formatted = self._transform_item(item, format_type)
            if formatted:
                all_items.append(formatted)

        # 3. Load Processed/Cleaned Data (e.g., Crisis Scenarios)
        logger.info("Loading Processed data...")
        processed_data = self.load_json_files(self.stage4_path)

        for item in processed_data:
            # Determine type of processed data
            # For now, simplistic handling based on content
            formatted = self._transform_generic(item, format_type)
            if formatted:
                all_items.append(formatted)

        logger.info(f"Synthesized {len(all_items)} total instruction items.")
        return all_items

    def _transform_item(self, item: dict, format_type: str) -> dict:
        """Transforms a specific therapy item into an instruction pair."""
        item_type = item.get("type", "")

        # Use a dictionary-based dispatch pattern to reduce branching
        transformers = {
            # CBT
            "CBT_Thought_Record": self._transform_cbt_thought_record,
            "CBT_Behavioral_Activation": self._transform_cbt_behavioral_activation,
            "CBT_Cognitive_Restructuring": self._transform_cbt_cognitive_restructuring,

            # DBT
            "DBT_Skill_Exercise": self._transform_dbt_skill_exercise,
            "DBT_Diary_Card": self._transform_dbt_diary_card,
            "DBT_Chain_Analysis": self._transform_dbt_chain_analysis,

            # EMDR
            "EMDR_Treatment_Plan": self._transform_emdr_treatment_plan,
            "EMDR_Phase_Protocol": self._transform_emdr_phase_protocol,
            "EMDR_Resource_Script": self._transform_emdr_resource_script,

            # ACT
            "ACT_Hexaflex_Exercise": self._transform_act_hexaflex_exercise,
            "ACT_Values_Clarification": self._transform_act_values_clarification,
            "ACT_Defusion_Technique": self._transform_act_defusion_technique,
        }

        transformer = transformers.get(item_type)
        if not transformer:
            return None  # Skip unsupported types for now

        instruction, input_text, output_text = transformer(item)
        return self._format_output(instruction, input_text, output_text, format_type)

    def _format_output(
        self, instruction: str, input_text: str, output_text: str, format_type: str
    ) -> dict:
        """Format the transformed data according to the specified format type."""
        if format_type == "alpaca":
            return to_alpaca(instruction, input_text, output_text)
        if format_type == "sharegpt":
            return to_sharegpt(
                self.DEFAULT_SYSTEM_PROMPT,
                f"{instruction}\n\n{input_text}",
                output_text
            )
        return None

    def _transform_cbt_thought_record(self, item: dict) -> tuple[str, str, str]:
        """Transform CBT Thought Record data."""
        data = item.get("data", {})
        instruction = (
            "Act as a CBT therapist. Complete a Thought Record "
            "for the client's situation."
        )
        input_text = (
            f"Situation: {data.get('situation')}\n"
            f"Emotion: {data.get('emotion', {}).get('label')}\n"
            f"Negative Thought: {data.get('automatic_thought')}"
        )
        output_text = (
            f"Cognitive Distortion: {data.get('cognitive_distortion')}\n"
            f"Balanced Thought: {data.get('balanced_thought', {}).get('content')}"
        )
        return instruction, input_text, output_text

    def _transform_cbt_behavioral_activation(self, item: dict) -> tuple[str, str, str]:
        """Transform CBT Behavioral Activation data."""
        data = item.get("data", {})
        instruction = "Create a Behavioral Activation log entry."
        input_text = (
            f"Activity: {data.get('activity')}\n"
            f"Metrics: Difficulty={data.get('metrics', {}).get('difficulty')}, "
            f"Pleasure={data.get('metrics', {}).get('pleasure')}"
        )
        output_text = f"Notes: {data.get('notes')}"
        return instruction, input_text, output_text

    def _transform_cbt_cognitive_restructuring(
        self, item: dict
    ) -> tuple[str, str, str]:
        """Transform CBT Cognitive Restructuring data."""
        data = item.get("data", {})
        instruction = (
            "Guide the client through cognitive restructuring "
            "for a negative thought."
        )
        input_text = f"Target Thought: {data.get('target_thought')}"
        questions = "\n".join([f"- {q}" for q in data.get("socratic_questions", [])])
        output_text = (
            f"Socratic Questions:\n{questions}\n\n"
            f"Alternative Perspective: {data.get('alternative_perspective')}"
        )
        return instruction, input_text, output_text

    def _transform_dbt_skill_exercise(self, item: dict) -> tuple[str, str, str]:
        """Transform DBT Skill Exercise data."""
        data = item.get("data", {})
        instruction = "Provide a DBT skill exercise instruction."
        input_text = f"Module: {data.get('module')}\nSkill: {data.get('skill')}"
        output_text = (
            f"Instructions: {data.get('instructions')}\n"
            f"Reflection: {', '.join(data.get('reflection_questions', []))}"
        )
        return instruction, input_text, output_text

    def _transform_dbt_diary_card(self, item: dict) -> tuple[str, str, str]:
        """Transform DBT Diary Card data."""
        data = item.get("data", {})
        urges = data.get("urges", {})
        emotions = data.get("emotions", {})
        instruction = "Summarize a DBT Diary Card entry."
        input_text = (
            f"Urges: Self-Harm={urges.get('self_harm')}, "
            f"Drug Use={urges.get('drug_use')}\n"
            f"Emotions: Sadness={emotions.get('sadness')}, "
            f"Anxiety={emotions.get('anxiety')}"
        )
        output_text = (
            f"Skills Used: {data.get('skills_used')}\n"
            f"Notes: {data.get('notes')}"
        )
        return instruction, input_text, output_text

    def _transform_dbt_chain_analysis(self, item: dict) -> tuple[str, str, str]:
        """Transform DBT Chain Analysis data."""
        data = item.get("data", {})
        chain = data.get("chain", {})
        instruction = "Perform a Chain Analysis for a problem behavior."
        input_text = (
            f"Problem Behavior: {data.get('problem_behavior')}\n"
            f"Vulnerability: {chain.get('vulnerability_factors')}"
        )
        output_text = (
            f"Prompting Event: {chain.get('prompting_event')}\n"
            f"Consequences: {chain.get('consequences', {}).get('immediate')}\n"
            f"Solution: "
            f"{data.get('solution_analysis', {}).get('skills_to_break_chain')}"
        )
        return instruction, input_text, output_text

    def _transform_emdr_treatment_plan(self, item: dict) -> tuple[str, str, str]:
        """Transform EMDR Treatment Plan data."""
        data = item.get("data", {})
        instruction = "Create an EMDR Targeting Sequence Plan."
        input_text = (
            f"Target Memory: {data.get('target_memory')}\n"
            f"Negative Cognition: {data.get('negative_cognition_nc')}"
        )
        output_text = f"Positive Cognition: {data.get('positive_cognition_pc')}"
        return instruction, input_text, output_text

    def _transform_emdr_phase_protocol(self, item: dict) -> tuple[str, str, str]:
        """Transform EMDR Phase Protocol data."""
        data = item.get("data", {})
        instruction = "Describe the steps for an EMDR Phase."
        input_text = f"Phase: {data.get('phase')} ({data.get('name')})"
        output_text = f"Focus: {data.get('focus')}\nSteps:\n" + "\n".join(
            [f"- {s}" for s in data.get("steps", [])]
        )
        return instruction, input_text, output_text

    def _transform_emdr_resource_script(self, item: dict) -> tuple[str, str, str]:
        """Transform EMDR Resource Script data."""
        data = item.get("data", {})
        instruction = "Provide a script for an EMDR resourcing technique."
        input_text = f"Technique: {data.get('technique')}"
        output_text = (
            f"Script Summary: {data.get('script_summary')}\n"
            f"BLS: {data.get('bls_instruction')}"
        )
        return instruction, input_text, output_text

    def _transform_act_hexaflex_exercise(self, item: dict) -> tuple[str, str, str]:
        """Transform ACT Hexaflex Exercise data."""
        data = item.get("data", {})
        instruction = "Suggest an ACT exercise for the specified process."
        input_text = f"Process: {data.get('process')}"
        output_text = (
            f"Exercise: {data.get('exercise_name')}\n"
            f"Instructions: {data.get('instructions')}"
        )
        return instruction, input_text, output_text

    def _transform_act_values_clarification(self, item: dict) -> tuple[str, str, str]:
        """Transform ACT Values Clarification data."""
        data = item.get("data", {})
        instruction = "Provide values clarification prompts for a specific domain."
        input_text = f"Domain: {data.get('domain')}"
        prompts = [f"- {p}" for p in data.get("prompts", [])]
        output_text = "Prompts:\n" + "\n".join(prompts)
        return instruction, input_text, output_text

    def _transform_act_defusion_technique(self, item: dict) -> tuple[str, str, str]:
        """Transform ACT Defusion Technique data."""
        data = item.get("data", {})
        instruction = "Suggest a cognitive defusion technique."
        input_text = f"Target Thought: {data.get('target_thought')}"
        output_text = (
            f"Technique: {data.get('technique')}\n"
            f"Instructions: {data.get('instructions')}"
        )
        return instruction, input_text, output_text

    def _transform_generic(self, item: dict, format_type: str) -> dict:
        """Handles generic cleaned data."""
        # Example handling for crisis scenarios if they adhere to a schema
        # For this example, assuming 'raw_text' -> 'scrubbed_content' structure
        # But our processed crisis scenarios might be in a specific format.
        # Let's check keys.

        if "content" in item and "risk_level" in item:
            # Crisis Scenario
            instruction = "Analyze the risk level of the following crisis scenario."
            input_text = item.get("content", "")
            output_text = f"Risk Level: {item.get('risk_level')}"

            if format_type == "alpaca":
                return to_alpaca(instruction, input_text, output_text)
            if format_type == "sharegpt":
                return to_sharegpt(
                    self.DEFAULT_SYSTEM_PROMPT, f"{instruction}\n\n{input_text}", output_text
                )

        return None

    def split_dataset(
        self, data: list[dict], train_ratio: float = 0.9, val_ratio: float = 0.05
    ) -> dict[str, list[dict]]:
        """Splits data into train/val/test."""
        random.shuffle(data)
        total = len(data)
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

        return {
            "train": data[:train_end],
            "val": data[train_end:val_end],
            "test": data[val_end:]
        }
