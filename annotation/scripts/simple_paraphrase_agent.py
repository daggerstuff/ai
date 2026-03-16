"""
Simple Therapeutic Conversation Paraphrasing Agent
Uses NVIDIA Nemotron with a more robust, text-based approach
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import openai


class SimpleParaphraser:
    """Generate paraphrased variations using text-based approach"""

    def __init__(self, model: str = "nvidia/llama-3.3-nemotron-super-49b-v1"):
        self.model = model
        self.client = openai.OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ.get("NVIDIA_API_KEY"),
        )

    def paraphrase_messages(
        self, messages: list[dict[str, Any]], variation_type: str = "lexical"
    ) -> list[dict[str, Any]]:
        """
        Paraphrase conversation messages

        Args:
            messages: List of conversation messages
            variation_type: Type of variation

        Returns:
            Paraphrased messages
        """

        # Convert messages to readable format
        conversation_text = self._messages_to_text(messages)

        # Get paraphrasing instructions
        instructions = self._get_instructions(variation_type)

        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"""You are a therapeutic conversation paraphraser.

{instructions}

IMPORTANT: Return the paraphrased conversation in the EXACT same format as the input.
Each line should start with either "User:" or "Assistant:" followed by the message.
Preserve the number of messages and the speaker roles.""",
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Paraphrase this conversation:\n\n{conversation_text}"
                        ),
                    },
                ],
                temperature=0.8,
                max_tokens=2000,
            )

            paraphrased_text = response.choices[0].message.content

            # Parse back to messages format
            paraphrased_messages = self._text_to_messages(paraphrased_text)

            # Validate we have the same number of messages
            if len(paraphrased_messages) != len(messages):
                print(
                    f"  Warning: Message count mismatch "
                    f"({len(paraphrased_messages)} vs {len(messages)}), "
                    f"using original"
                )
                return messages

            return paraphrased_messages

        except Exception as e:
            print(f"  Error during paraphrasing: {e}, using original")
            return messages

    def _messages_to_text(self, messages: list[dict[str, Any]]) -> str:
        """Convert messages to readable text format"""
        lines = []
        for msg in messages:
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n\n".join(lines)

    def _text_to_messages(self, text: str) -> list[dict[str, Any]]:
        """Convert text back to messages format"""
        messages = []

        # Split by double newlines or single newlines with role prefixes
        lines = text.strip().split("\n")

        current_role = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if line starts with a role
            if line.lower().startswith("user:"):
                # Save previous message if exists
                if current_role and current_content:
                    messages.append(
                        {"role": current_role, "content": " ".join(current_content)}
                    )
                current_role = "user"
                current_content = [line[5:].strip()]
            elif line.lower().startswith("assistant:"):
                # Save previous message if exists
                if current_role and current_content:
                    messages.append(
                        {"role": current_role, "content": " ".join(current_content)}
                    )
                current_role = "assistant"
                current_content = [line[10:].strip()]
            else:
                # Continuation of previous message
                if current_content:
                    current_content.append(line)

        # Save last message
        if current_role and current_content:
            messages.append(
                {"role": current_role, "content": " ".join(current_content)}
            )

        return messages

    def _get_instructions(self, variation_type: str) -> str:
        """Get paraphrasing instructions for variation type"""

        instructions = {
            "lexical": """VARIATION: Lexical (Synonym Substitution)
- Replace words with synonyms
- Use alternative phrasings
- Keep sentence structure similar
- Maintain emotional intensity

Example:
Original: "I'm struggling with anxiety"
Paraphrased: "I'm having difficulty with nervousness"
""",
            "syntactic": """VARIATION: Syntactic (Sentence Restructuring)
- Rearrange sentence components
- Change passive ↔ active voice
- Modify clause order
- Keep similar words

Example:
Original: "I can't sleep because I'm anxious"
Paraphrased: "My anxiety is keeping me awake"
""",
            "stylistic_formal": """VARIATION: Stylistic (Formal)
- Use professional, clinical language
- Avoid contractions (I'm → I am)
- Use complete sentences

Example:
Original: "I'm really stressed out"
Paraphrased: "I am experiencing significant distress"
""",
            "stylistic_informal": """VARIATION: Stylistic (Informal)
- Use casual, conversational language
- Include contractions
- Use colloquialisms

Example:
Original: "I am experiencing distress"
Paraphrased: "I'm really stressed out"
""",
        }

        return instructions.get(variation_type, instructions["lexical"])

    def generate_variations(
        self,
        source_file: str,
        output_dir: str,
        variation_types: list[str],
        per_sample: int = 1,
    ) -> None:
        """Generate paraphrased variations"""

        source_path = Path(source_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load source conversations
        conversations = []
        with open(source_path) as f:
            for line in f:
                if line.strip():
                    conversations.append(json.loads(line))

        print(f"📚 Loaded {len(conversations)} conversations")
        print(f"🎯 Generating {len(variation_types)} variation types")
        total_target = len(conversations) * len(variation_types) * per_sample
        print(f"📊 Total target: {total_target} paraphrases\n")

        total_generated = 0
        total_successful = 0

        for var_type in variation_types:
            var_dir = output_path / var_type
            var_dir.mkdir(exist_ok=True)

            print(f"🔄 Generating {var_type} variations...")

            for i, conv in enumerate(conversations):
                task_id = conv.get("task_id", f"task_{i}")

                for j in range(per_sample):
                    try:
                        # Get original messages (handle both formats)
                        original_messages = conv.get("messages", [])
                        if not original_messages and "data" in conv:
                            original_messages = conv.get("data", {}).get("messages", [])

                        if not original_messages:
                            print(f"  ⚠️  No messages in {task_id}, skipping")
                            continue

                        # Paraphrase messages
                        paraphrased_messages = self.paraphrase_messages(
                            original_messages, var_type
                        )

                        # Create new conversation
                        paraphrased = conv.copy()
                        paraphrased["task_id"] = f"{task_id}_{var_type}_{j}"
                        paraphrased["messages"] = paraphrased_messages
                        paraphrased["metadata"] = paraphrased.get("metadata", {}).copy()
                        paraphrased["metadata"]["variation_type"] = var_type
                        paraphrased["metadata"]["paraphrase_model"] = self.model
                        paraphrased["metadata"]["original_task_id"] = task_id

                        # Save
                        output_file = var_dir / f"{task_id}_{var_type}_{j}.jsonl"
                        with open(output_file, "w") as f:
                            f.write(json.dumps(paraphrased) + "\n")

                        total_generated += 1
                        if len(paraphrased_messages) == len(original_messages):
                            total_successful += 1

                        if total_generated % 10 == 0:
                            print(
                                f"  ✅ Generated {total_generated} "
                                f"({total_successful} successful)..."
                            )

                    except Exception as e:
                        print(f"  ❌ Error with {task_id} ({var_type}): {e}")

            print(f"  ✓ Completed {var_type}: {total_generated} total\n")

        print("\n🎉 Complete!")
        print(f"📊 Total generated: {total_generated}")
        print(f"✅ Successful: {total_successful}")
        print(f"⚠️  Fallback to original: {total_generated - total_successful}")

        # Combine all variations
        combined_file = output_path / "all_paraphrases.jsonl"
        with open(combined_file, "w") as out_f:
            for var_type in variation_types:
                var_dir = output_path / var_type
                for jsonl_file in sorted(var_dir.glob("*.jsonl")):
                    with open(jsonl_file) as in_f:
                        out_f.write(in_f.read())

        print(f"📦 Combined file: {combined_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate paraphrased variations (simple text-based approach)"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--variations", default="lexical,syntactic", help="Variation types"
    )
    parser.add_argument(
        "--per-sample", type=int, default=1, help="Variations per sample"
    )
    parser.add_argument(
        "--model", default="nvidia/llama-3.3-nemotron-super-49b-v1", help="Model"
    )

    args = parser.parse_args()

    variation_types = [v.strip() for v in args.variations.split(",")]

    paraphraser = SimpleParaphraser(model=args.model)
    paraphraser.generate_variations(
        source_file=args.input,
        output_dir=args.output,
        variation_types=variation_types,
        per_sample=args.per_sample,
    )


if __name__ == "__main__":
    main()
