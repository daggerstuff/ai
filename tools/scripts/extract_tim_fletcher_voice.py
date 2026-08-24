#!/usr/bin/env python3
"""
Tim Fletcher Voice Extraction System

Analyzes 913 YouTube transcripts to extract Tim Fletcher's:
- Teaching style and flow
- Personality traits
- Way of explaining complex trauma concepts
- Analogies and examples
- Sentence structures and patterns
- Vocabulary and phrasing
"""

import json
import logging
import os
import random
import re
from collections import Counter
from pathlib import Path

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MIN_SENTENCE_WORDS = 2
MAX_MARKED_SENTENCE_LENGTH = 200
MIN_COMMON_PHRASE_COUNT = 50


class TimFletcherVoiceExtractor:
    def __init__(self, transcripts_dir: str = ".notes/transcripts"):
        self.transcripts_dir = Path(transcripts_dir)
        self.output_dir = Path("ai/data/tim_fletcher_voice")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dotenv_values = self._load_dotenv()
        self.nim_api_key = self._resolve_env("NVIDIA_NIM_API_KEY") or self._resolve_env("NVIDIA_API_KEY")
        self.nim_base_url = self._resolve_env("NVIDIA_NIM_BASE_URL") or "https://integrate.api.nvidia.com/v1"
        model_from_env = self._resolve_env("TIM_FLETCHER_NIM_MODEL") or self._resolve_env("NVIDIA_NIM_MODEL")
        self.openai_model = model_from_env or os.getenv(
            "OPENAI_MODEL",
            "meta/llama-3.1-8b-instruct",
        )
        self.openai_client = self._init_openai_client()

        self.voice_profile = {
            "common_phrases": Counter(),
            "sentence_starters": Counter(),
            "analogies": [],
            "teaching_patterns": [],
            "examples": [],
            "transition_phrases": Counter(),
            "empathy_markers": Counter(),
            "explanation_structures": [],
        }

    def extract_voice_patterns(self) -> dict:
        """Extract Tim Fletcher's voice patterns from all transcripts"""
        logger.info(f"🎙️ Analyzing transcripts from {self.transcripts_dir}")

        transcript_files = list(self.transcripts_dir.glob("*.txt"))
        logger.info(f"📁 Found {len(transcript_files)} transcript files")

        all_text = []
        for i, transcript_file in enumerate(transcript_files, 1):
            if i % 100 == 0:
                logger.info(f"   Processing transcript {i}/{len(transcript_files)}")

            try:
                with open(transcript_file, encoding="utf-8") as f:
                    text = f.read()
                    all_text.append(text)
                    self._analyze_transcript(text)
            except Exception as e:
                logger.error(f"Error reading {transcript_file}: {e}")

        # Analyze combined patterns
        combined_text = "\n\n".join(all_text)
        self._extract_teaching_style(combined_text)

        logger.info("✅ Voice pattern extraction complete")
        return self.voice_profile

    def _analyze_transcript(self, text: str):
        """Analyze a single transcript for voice patterns"""
        # Extract sentences
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in sentences:
            # Sentence starters (first 2-3 words)
            words = sentence.split()
            if len(words) >= MIN_SENTENCE_WORDS:
                starter = " ".join(words[:2])
                self.voice_profile["sentence_starters"][starter] += 1

            # Transition phrases
            transitions = [
                "And so",
                "Now",
                "So",
                "But",
                "And then",
                "What happens",
                "Let me",
                "Think about",
                "Imagine",
                "What I find",
                "One of the things",
                "The reality is",
                "What we see",
            ]
            for transition in transitions:
                if sentence.lower().startswith(transition.lower()):
                    self.voice_profile["transition_phrases"][transition] += 1

            # Empathy markers
            empathy_patterns = [
                "I understand",
                "I know",
                "I get it",
                "That's painful",
                "That's hard",
                "You might feel",
                "Many people",
                "Some of you",
                "For many",
                "What you're going through",
            ]
            for pattern in empathy_patterns:
                if pattern.lower() in sentence.lower():
                    self.voice_profile["empathy_markers"][pattern] += 1

            # Analogies (look for "like", "as if", "imagine")
            if (
                any(marker in sentence.lower() for marker in ["like a", "as if", "imagine", "think of"])
                and len(sentence) < MAX_MARKED_SENTENCE_LENGTH  # Keep reasonable length
            ):
                self.voice_profile["analogies"].append(sentence)

            # Examples (look for "let's say", "for example")
            if (
                any(marker in sentence.lower() for marker in ["let's say", "for example", "think back to"])
                and len(sentence) < MAX_MARKED_SENTENCE_LENGTH
            ):
                self.voice_profile["examples"].append(sentence)

    def _extract_teaching_style(self, text: str):
        """Extract high-level teaching style patterns"""
        # Common multi-word phrases
        words = text.lower().split()
        for i in range(len(words) - 2):
            phrase = " ".join(words[i : i + 3])
            self.voice_profile["common_phrases"][phrase] += 1

        # Teaching patterns (how he structures explanations)
        patterns = [
            "First",
            "Second",
            "Third",  # Numbered points
            "What happens is",
            "The reality is",
            "What we find",
            "One of the key",
            "It's important to understand",
            "Let me give you an example",
            "Think about this",
        ]

        for pattern in patterns:
            count = text.lower().count(pattern.lower())
            if count > 0:
                self.voice_profile["teaching_patterns"].append({"pattern": pattern, "frequency": count})

    def generate_voice_profile_report(self) -> str:
        """Generate a comprehensive voice profile report"""
        report = []
        report.append("# Tim Fletcher Voice Profile\n")
        report.append("**Analyzed**: 913 YouTube transcripts on complex trauma, PTSD, recovery\n\n")

        # Top sentence starters
        report.append("## Top Sentence Starters\n")
        report.append("How Tim Fletcher typically begins his sentences:\n\n")
        for starter, count in self.voice_profile["sentence_starters"].most_common(20):
            report.append(f'- **"{starter}..."** ({count} times)\n')

        # Top transitions
        report.append("\n## Transition Phrases\n")
        report.append("How Tim connects ideas and moves between topics:\n\n")
        for transition, count in self.voice_profile["transition_phrases"].most_common(15):
            report.append(f'- **"{transition}"** ({count} times)\n')

        # Empathy markers
        report.append("\n## Empathy & Connection Markers\n")
        report.append("How Tim shows understanding and connects with audience:\n\n")
        for marker, count in self.voice_profile["empathy_markers"].most_common(15):
            report.append(f'- **"{marker}"** ({count} times)\n')

        # Sample analogies
        report.append("\n## Sample Analogies & Metaphors\n")
        report.append("Tim's way of explaining complex concepts:\n\n")
        for analogy in self.voice_profile["analogies"][:10]:
            report.append(f"- {analogy}\n")

        # Sample examples
        report.append("\n## Sample Examples\n")
        report.append("How Tim illustrates points with examples:\n\n")
        for example in self.voice_profile["examples"][:10]:
            report.append(f"- {example}\n")

        # Common phrases
        report.append("\n## Most Common 3-Word Phrases\n")
        for phrase, count in self.voice_profile["common_phrases"].most_common(30):
            if count > MIN_COMMON_PHRASE_COUNT:  # Only very common phrases
                report.append(f'- "{phrase}" ({count} times)\n')

        return "".join(report)

    def save_voice_profile(self):
        """Save voice profile data to files"""
        # Save JSON data
        profile_data = {
            "sentence_starters": dict(self.voice_profile["sentence_starters"].most_common(50)),
            "transition_phrases": dict(self.voice_profile["transition_phrases"].most_common(30)),
            "empathy_markers": dict(self.voice_profile["empathy_markers"].most_common(30)),
            "common_phrases": dict(self.voice_profile["common_phrases"].most_common(100)),
            "analogies": self.voice_profile["analogies"][:50],
            "examples": self.voice_profile["examples"][:50],
            "teaching_patterns": self.voice_profile["teaching_patterns"],
        }

        profile_file = self.output_dir / "tim_fletcher_voice_profile.json"
        with open(profile_file, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved voice profile to {profile_file}")

        # Save markdown report
        report = self.generate_voice_profile_report()
        report_file = self.output_dir / "tim_fletcher_voice_analysis.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"📝 Saved voice analysis report to {report_file}")

    def generate_synthetic_conversations(self, num_conversations: int = 1000) -> list[dict]:
        """Generate synthetic therapeutic conversations in Tim Fletcher's voice"""
        logger.info(f"🎨 Generating {num_conversations} synthetic conversations...")

        # This will be implemented with an LLM API
        # For now, create a template structure
        conversations = []

        # Load voice profile
        profile_file = self.output_dir / "tim_fletcher_voice_profile.json"
        if not profile_file.exists():
            logger.warning("Voice profile not found. Run extraction first.")
            return []

        with open(profile_file) as f:
            profile = json.load(f)

        # Create conversation generation instructions
        generation_prompt = self._create_generation_prompt(profile)

        prompt_file = self.output_dir / "conversation_generation_prompt.txt"
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(generation_prompt)
        logger.info(f"📋 Saved generation prompt to {prompt_file}")

        topics = [
            "Complex trauma, hypervigilance, and nervous system regulation",
            "PTSD triggers and coping mechanisms in daily life",
            "Rebuilding trust in relationships after emotional betrayal",
            "Managing shame and self-criticism during recovery",
            "Sleep disruption after trauma and practical grounding",
            "Body-based regulation when panic spikes",
            "Boundaries, rupture, and reparation in attachment wounds",
            "Dissociation and returning to the present moment",
            "Trauma recovery progress without emotional overwhelm",
            "Integrating therapy concepts into relationships and work",
        ]

        max_requests = int(os.getenv("TIM_FLETCHER_MAX_SYNTHETIC", "50"))
        target_conversations = min(num_conversations, max_requests)
        if num_conversations > target_conversations:
            logger.warning(
                "🔧 Limiting synthetic conversation generation to %s to avoid runaway API spend.",
                target_conversations,
            )

        for i in range(target_conversations):
            topic = topics[i % len(topics)]
            topic_prompt = (
                f"{generation_prompt}\n\n"
                f"### Topic\n- {topic}\n\n"
                f"- {topic}\n\n"
                "Return ONLY a single JSON object matching the documented schema."
            )

            if self.openai_client is None:
                conversations.append(self._build_fallback_conversation(i + 1, topic, profile))
                continue

            try:
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a senior trauma-informed therapist writer.",
                        },
                        {"role": "user", "content": topic_prompt},
                    ],
                    temperature=0.7,
                    max_tokens=1500,
                )
                raw = response.choices[0].message.content if response.choices else ""
                parsed = self._extract_json_from_response(raw or "")
                if self._is_valid_conversation_payload(parsed):
                    parsed["metadata"]["topic"] = topic
                    parsed["metadata"]["index"] = i + 1
                    conversations.append(parsed)
                    continue
            except Exception as exc:
                logger.error("OpenAI generation failed for topic '%s': %s", topic, exc)

            conversations.append(self._build_fallback_conversation(i + 1, topic, profile))

        logger.info(f"✅ Generated {len(conversations)} synthetic conversations")
        return conversations

    def _extract_json_from_response(self, content: str) -> dict:
        """Extract JSON object from a model response."""
        try:
            return json.loads(content.strip())
        except Exception:
            pass

        json_block = re.search(r"```json\s*(.*?)\s*```", content, flags=re.DOTALL | re.IGNORECASE)
        if not json_block:
            return {}
        try:
            return json.loads(json_block.group(1).strip())
        except Exception:
            return {}

    def _resolve_env(self, key: str) -> str | None:
        """Resolve an environment value from loaded .env file or runtime env."""
        if key in self.dotenv_values:
            return self.dotenv_values[key]
        return os.getenv(key)

    @staticmethod
    def _load_dotenv() -> dict:
        """Load a simple .env file without external dependencies."""
        env_path = Path(".env")
        values: dict[str, str] = {}

        if not env_path.exists():
            return values

        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :].strip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip().strip('"').strip("'")
                if key:
                    values[key] = value
        except OSError as exc:
            logger.warning("⚠️ Unable to read .env file: %s", exc)

        return values

    def _is_valid_conversation_payload(self, payload: dict) -> bool:
        """Validate minimum schema for generated conversation."""
        if not isinstance(payload, dict):
            return False
        conversation = payload.get("conversation")
        metadata = payload.get("metadata")
        return (
            isinstance(conversation, list)
            and conversation
            and isinstance(metadata, dict)
            and all(
                isinstance(turn, dict)
                and turn.get("role") in {"client", "therapist"}
                and isinstance(turn.get("content"), str)
                and bool(turn["content"].strip())
                for turn in conversation
            )
        )

    def _build_fallback_conversation(self, index: int, topic: str, profile: dict) -> dict:
        starters = list(profile.get("sentence_starters", {}).keys()) or ["Let's take this step by step"]
        transitions = list(profile.get("transition_phrases", {}).keys()) or ["Now", "Let's look at this"]
        empathy = list(profile.get("empathy_markers", {}).keys()) or ["I understand", "That sounds hard"]
        analogy = random.choice(profile.get("analogies") or ["Like retraining a nervous system takes time."])
        example = random.choice(
            profile.get("examples") or ["Imagine your alarm system is extra sensitive for a while."]
        )

        conversation = [
            {"role": "client", "content": f"I feel overwhelmed around this topic. This topic is: {topic}."},
            {
                "role": "therapist",
                "content": (
                    f"{random.choice(starters)}... {random.choice(empathy)}. "
                    f"{random.choice(transitions)} first, let's break this down: "
                    "This is about noticing your body first, then naming the moment, "
                    "then choosing one small action."
                ),
            },
            {
                "role": "client",
                "content": "That makes sense, but I get stuck before I can do those steps.",
            },
            {
                "role": "therapist",
                "content": (
                    f"{random.choice(transitions)} I hear you. "
                    f"{random.choice(empathy)}. Think of it like {analogy.lower()} "
                    f"and for example {example.lower()}"
                ),
            },
            {
                "role": "client",
                "content": "So what should I do when I feel stuck again?",
            },
            {
                "role": "therapist",
                "content": (
                    "Great question. First, pause and do one 60-second regulation breath. "
                    "Second, ground with three things you can see, touch, and hear. "
                    "Third, tell yourself: this is activation, and it can pass without me doing anything."
                ),
            },
        ]
        return {
            "conversation": conversation,
            "metadata": {
                "source": "tim_fletcher_synthetic",
                "topic": topic,
                "index": index,
                "mode": "fallback" if self.openai_client is None else "openai_parsing_failed",
            },
        }

    def _init_openai_client(self):
        if not OPENAI_AVAILABLE or not self.nim_api_key:
            return None

        if not self.nim_base_url:
            logger.warning("⚠️ NVIDIA NIM base URL not configured; defaulting to https://integrate.api.nvidia.com/v1")
            self.nim_base_url = "https://integrate.api.nvidia.com/v1"

        if not self.nim_api_key:
            logger.warning("⚠️ NVIDIA NIM API key not configured; synthetic generation will fall back to templates.")
            return None

        try:
            return OpenAI(api_key=self.nim_api_key, base_url=self.nim_base_url)
        except TypeError:
            # Defensive fallback for older OpenAI clients not accepting base_url kwarg.
            return OpenAI(api_key=self.nim_api_key)

    def _create_generation_prompt(self, profile: dict) -> str:
        """Create a prompt for generating conversations in Tim's voice"""
        prompt = """# Generate Therapeutic Conversations in Tim Fletcher's Voice

## Voice Characteristics

### Sentence Starters (use these frequently):
"""
        for starter, _count in list(profile["sentence_starters"].items())[:15]:
            prompt += f'- "{starter}..."\n'

        prompt += "\n### Transition Phrases:\n"
        for phrase, _count in list(profile["transition_phrases"].items())[:10]:
            prompt += f'- "{phrase}"\n'

        prompt += "\n### Empathy Markers:\n"
        for marker, _count in list(profile["empathy_markers"].items())[:10]:
            prompt += f'- "{marker}"\n'

        prompt += "\n### Teaching Style:\n"
        prompt += "- Use numbered points (First, Second, Third)\n"
        prompt += "- Give concrete examples starting with 'Let's say' or 'Think about'\n"
        prompt += "- Use analogies with 'It's like' or 'Imagine'\n"
        prompt += "- Break down complex concepts into simple steps\n"
        prompt += "- Connect to real-life scenarios\n"
        prompt += "- Show deep empathy and understanding\n"
        prompt += "- Normalize the client's experience with 'Many people' or 'For some people'\n"

        prompt += "\n### Sample Analogies:\n"
        for analogy in profile["analogies"][:5]:
            prompt += f"- {analogy}\n"

        prompt += """

## Generation Task

Generate therapeutic conversations where the therapist speaks in Tim Fletcher's voice.

**Format**:
```json
{
  "conversation": [
    {"role": "client", "content": "..."},
    {"role": "therapist", "content": "..."}
  ],
  "metadata": {
    "source": "tim_fletcher_synthetic",
    "topic": "complex_trauma/ptsd/recovery/etc"
  }
}
```

**Requirements**:
1. Therapist responses must use Tim's sentence starters, transitions, and empathy markers
2. Include analogies and examples in his style
3. Break down complex concepts step-by-step
4. Show deep understanding and normalization
5. Multi-turn conversations (4-8 exchanges)
6. Focus on complex trauma, PTSD, recovery topics
"""

        return prompt


def main():
    logger.info("🚀 Tim Fletcher Voice Extraction System")
    logger.info("=" * 60)

    extractor = TimFletcherVoiceExtractor()

    # Extract voice patterns
    extractor.extract_voice_patterns()

    # Save results
    extractor.save_voice_profile()

    # Generate conversation template
    extractor.generate_synthetic_conversations()

    logger.info("\n✅ Voice extraction complete!")
    logger.info(f"📁 Output directory: {extractor.output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Review voice profile: tim_fletcher_voice_profile.json")
    logger.info("2. Review analysis report: tim_fletcher_voice_analysis.md")
    logger.info("3. Use conversation_generation_prompt.txt with LLM API to generate synthetic conversations")


if __name__ == "__main__":
    main()
