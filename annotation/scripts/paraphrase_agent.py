"""
Therapeutic Conversation Paraphrasing Agent
Uses NVIDIA Nemotron to generate diverse variations while preserving
emotional/crisis labels
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import openai


class TherapeuticParaphraser:
    """Generate paraphrased variations of therapeutic conversations"""

    def __init__(self, model: str = "nvidia/llama-3.3-nemotron-super-49b-v1"):
        self.model = model
        self.client = openai.OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ.get("NVIDIA_API_KEY"),
        )

    def paraphrase_conversation(
        self, conversation: dict[str, Any], variation_type: str = "lexical"
    ) -> dict[str, Any]:
        """
        Generate paraphrased variation of conversation

        Args:
            conversation: Original conversation with messages
            variation_type: Type of variation
                (lexical, syntactic, stylistic_formal, etc.)

        Returns:
            Paraphrased conversation with same structure
        """

        system_prompt = self._get_paraphrase_prompt(variation_type)

        # Extract just the messages for paraphrasing
        messages = conversation.get("messages", [])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Paraphrase this conversation:\n\n"
                        f"{json.dumps(messages, indent=2)}"
                    ),
                },
            ],
            temperature=0.8,  # Higher for diversity
            max_tokens=2000,
        )

        # Parse the paraphrased messages
        try:
            paraphrased_messages = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # If model didn't return valid JSON, use original
            print(f"Warning: Failed to parse paraphrase for variation {variation_type}")
            paraphrased_messages = messages

        # Create new conversation with paraphrased messages
        paraphrased = conversation.copy()
        paraphrased["messages"] = paraphrased_messages
        paraphrased["metadata"] = paraphrased.get("metadata", {})
        paraphrased["metadata"]["variation_type"] = variation_type
        paraphrased["metadata"]["paraphrase_model"] = self.model

        return paraphrased

    def _get_paraphrase_prompt(self, variation_type: str) -> str:
        """Get system prompt for specific variation type"""

        base = """You are a therapeutic conversation paraphraser.

Your task is to rewrite the conversation while:
1. Preserving the emotional tone and crisis level
2. Maintaining the therapeutic context
3. Ensuring natural, authentic dialogue
4. Keeping the same number of messages
5. Preserving the role of each speaker (user/assistant)

Return ONLY a valid JSON array of messages in the same format as the input.
Each message should have "role" and "content" fields.
"""

        variations = {
            "lexical": """
VARIATION TYPE: Lexical (Synonym Substitution)
- Replace words with synonyms
- Use alternative phrasings
- Keep sentence structure similar
- Maintain emotional intensity

Example:
Original: "I'm struggling with anxiety"
Paraphrased: "I'm having difficulty with nervousness"
""",
            "syntactic": """
VARIATION TYPE: Syntactic (Sentence Restructuring)
- Rearrange sentence components
- Change passive ↔ active voice
- Modify clause order
- Keep the same words where possible

Example:
Original: "I can't sleep because I'm anxious"
Paraphrased: "My anxiety is keeping me awake"
""",
            "stylistic_formal": """
VARIATION TYPE: Stylistic (Formal)
- Use professional, clinical language
- Avoid contractions (I'm → I am)
- Use complete sentences
- Maintain therapeutic appropriateness

Example:
Original: "I'm really stressed out"
Paraphrased: "I am experiencing significant distress"
""",
            "stylistic_informal": """
VARIATION TYPE: Stylistic (Informal)
- Use casual, conversational language
- Include contractions
- Use colloquialisms where appropriate
- Keep it natural and relatable

Example:
Original: "I am experiencing distress"
Paraphrased: "I'm really stressed out"
""",
            "contextual": """
VARIATION TYPE: Contextual (Scenario Change)
- Change the specific scenario (work → family, relationships → health)
- Preserve the emotional pattern and intensity
- Keep the crisis level identical
- Maintain the therapeutic dynamic

Example:
Original: "My boss is making me anxious"
Paraphrased: "My family situation is making me anxious"
""",
        }

        return base + "\n\n" + variations.get(variation_type, variations["lexical"])

    def generate_variations(
        self,
        source_file: str,
        output_dir: str,
        variation_types: list[str],
        per_sample: int = 1,
    ) -> None:
        """
        Generate paraphrased variations for all conversations in source file

        Args:
            source_file: Path to JSONL file with annotated conversations
            output_dir: Directory to save paraphrased variations
            variation_types: List of variation types to generate
            per_sample: Number of variations per type per sample
        """

        source_path = Path(source_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load source conversations
        conversations = []
        with open(source_path) as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))

        print(f"📚 Loaded {len(conversations)} conversations from {source_file}")
        print(f"🎯 Generating {len(variation_types)} variation types")
        print(f"🔢 {per_sample} variations per type per sample")
        total_output = len(conversations) * len(variation_types) * per_sample
        print(f"📊 Total output: {total_output} paraphrases\n")

        total_generated = 0

        for var_type in variation_types:
            var_dir = output_path / var_type
            var_dir.mkdir(exist_ok=True)

            print(f"🔄 Generating {var_type} variations...")

            for i, conv in enumerate(conversations):
                task_id = conv.get("task_id", f"task_{i}")

                for j in range(per_sample):
                    try:
                        # Generate paraphrase
                        paraphrased = self.paraphrase_conversation(conv, var_type)

                        # Update task_id to reflect variation
                        paraphrased["task_id"] = f"{task_id}_{var_type}_{j}"

                        # Save to file
                        output_file = var_dir / f"{task_id}_{var_type}_{j}.jsonl"
                        with open(output_file, "w") as f:
                            f.write(json.dumps(paraphrased) + "\n")

                        total_generated += 1

                        if (total_generated) % 10 == 0:
                            print(f"  ✅ Generated {total_generated} paraphrases...")

                    except Exception as e:
                        print(f"  ❌ Error paraphrasing {task_id} ({var_type}): {e}")

        print(f"\n🎉 Complete! Generated {total_generated} paraphrases")
        print(f"📁 Saved to: {output_path}")

        # Combine all variations into single file
        combined_file = output_path / "all_paraphrases.jsonl"
        with open(combined_file, "w") as out_f:
            for var_type in variation_types:
                var_dir = output_path / var_type
                for jsonl_file in var_dir.glob("*.jsonl"):
                    with open(jsonl_file) as in_f:
                        out_f.write(in_f.read())

        print(f"📦 Combined file: {combined_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate paraphrased variations of therapeutic conversations"
    )
    parser.add_argument(
        "--input", required=True, help="Input JSONL file with annotated conversations"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for paraphrased variations",
    )
    parser.add_argument(
        "--variations",
        default="lexical,syntactic",
        help="Comma-separated list of variation types",
    )
    parser.add_argument(
        "--per-sample",
        type=int,
        default=1,
        help="Number of variations per type per sample",
    )
    parser.add_argument(
        "--model",
        default="nvidia/llama-3.3-nemotron-super-49b-v1",
        help="NVIDIA model to use for paraphrasing",
    )

    args = parser.parse_args()

    # Parse variation types
    variation_types = [v.strip() for v in args.variations.split(",")]

    # Create paraphraser
    paraphraser = TherapeuticParaphraser(model=args.model)

    # Generate variations
    paraphraser.generate_variations(
        source_file=args.input,
        output_dir=args.output,
        variation_types=variation_types,
        per_sample=args.per_sample,
    )


if __name__ == "__main__":
    main()
