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
import re
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TimFletcherVoiceExtractor:
    def __init__(self, transcripts_dir: str = ".notes/transcripts"):
        self.transcripts_dir = Path(transcripts_dir)
        self.output_dir = Path("ai/data/tim_fletcher_voice")
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def extract_voice_patterns(self) -> Dict:
        """Extract Tim Fletcher's voice patterns from all transcripts"""
        logger.info(f"🎙️ Analyzing transcripts from {self.transcripts_dir}")

        transcript_files = list(self.transcripts_dir.glob("*.txt"))
        logger.info(f"📁 Found {len(transcript_files)} transcript files")

        all_text = []
        for i, transcript_file in enumerate(transcript_files, 1):
            if i % 100 == 0:
                logger.info(f"   Processing transcript {i}/{len(transcript_files)}")

            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
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
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in sentences:
            # Sentence starters (first 2-3 words)
            words = sentence.split()
            if len(words) >= 2:
                starter = ' '.join(words[:2])
                self.voice_profile["sentence_starters"][starter] += 1

            # Transition phrases
            transitions = [
                "And so", "Now", "So", "But", "And then", "What happens",
                "Let me", "Think about", "Imagine", "What I find",
                "One of the things", "The reality is", "What we see"
            ]
            for transition in transitions:
                if sentence.lower().startswith(transition.lower()):
                    self.voice_profile["transition_phrases"][transition] += 1

            # Empathy markers
            empathy_patterns = [
                "I understand", "I know", "I get it", "That's painful",
                "That's hard", "You might feel", "Many people",
                "Some of you", "For many", "What you're going through"
            ]
            for pattern in empathy_patterns:
                if pattern.lower() in sentence.lower():
                    self.voice_profile["empathy_markers"][pattern] += 1

            # Analogies (look for "like", "as if", "imagine")
            if any(marker in sentence.lower() for marker in ["like a", "as if", "imagine", "think of"]):
                if len(sentence) < 200:  # Keep reasonable length
                    self.voice_profile["analogies"].append(sentence)

            # Examples (look for "let's say", "for example")
            if any(marker in sentence.lower() for marker in ["let's say", "for example", "think back to"]):
                if len(sentence) < 200:
                    self.voice_profile["examples"].append(sentence)

    def _extract_teaching_style(self, text: str):
        """Extract high-level teaching style patterns"""
        # Common multi-word phrases
        words = text.lower().split()
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i + 3])
            self.voice_profile["common_phrases"][phrase] += 1

        # Teaching patterns (how he structures explanations)
        patterns = [
            "First", "Second", "Third",  # Numbered points
            "What happens is", "The reality is", "What we find",
            "One of the key", "It's important to understand",
            "Let me give you an example", "Think about this",
        ]

        for pattern in patterns:
            count = text.lower().count(pattern.lower())
            if count > 0:
                self.voice_profile["teaching_patterns"].append({
                    "pattern": pattern,
                    "frequency": count
                })

    def generate_voice_profile_report(self) -> str:
        """Generate a comprehensive voice profile report"""
        report = []
        report.append("# Tim Fletcher Voice Profile\n")
        report.append(f"**Analyzed**: 913 YouTube transcripts on complex trauma, PTSD, recovery\n\n")

        # Top sentence starters
        report.append("## Top Sentence Starters\n")
        report.append("How Tim Fletcher typically begins his sentences:\n\n")
        for starter, count in self.voice_profile["sentence_starters"].most_common(20):
            report.append(f"- **\"{starter}...\"** ({count} times)\n")

        # Top transitions
        report.append("\n## Transition Phrases\n")
        report.append("How Tim connects ideas and moves between topics:\n\n")
        for transition, count in self.voice_profile["transition_phrases"].most_common(15):
            report.append(f"- **\"{transition}\"** ({count} times)\n")

        # Empathy markers
        report.append("\n## Empathy & Connection Markers\n")
        report.append("How Tim shows understanding and connects with audience:\n\n")
        for marker, count in self.voice_profile["empathy_markers"].most_common(15):
            report.append(f"- **\"{marker}\"** ({count} times)\n")

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
            if count > 50:  # Only very common phrases
                report.append(f"- \"{phrase}\" ({count} times)\n")

        return ''.join(report)

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
        with open(profile_file, 'w', encoding='utf-8') as f:
            json.dump(profile_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved voice profile to {profile_file}")

        # Save markdown report
        report = self.generate_voice_profile_report()
        report_file = self.output_dir / "tim_fletcher_voice_analysis.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"📝 Saved voice analysis report to {report_file}")

    def generate_synthetic_conversations(self, num_conversations: int = 1000) -> List[Dict]:
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

        with open(profile_file, 'r') as f:
            profile = json.load(f)

        # Create conversation generation instructions
        generation_prompt = self._create_generation_prompt(profile)

        prompt_file = self.output_dir / "conversation_generation_prompt.txt"
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(generation_prompt)
        logger.info(f"📋 Saved generation prompt to {prompt_file}")

        # TODO: Integrate with OpenAI API to generate conversations
        # For now, return empty list
        return conversations

    def _create_generation_prompt(self, profile: Dict) -> str:
        """Create a prompt for generating conversations in Tim's voice"""
        prompt = """# Generate Therapeutic Conversations in Tim Fletcher's Voice

## Voice Characteristics

### Sentence Starters (use these frequently):
"""
        for starter, count in list(profile["sentence_starters"].items())[:15]:
            prompt += f"- \"{starter}...\"\n"

        prompt += "\n### Transition Phrases:\n"
        for phrase, count in list(profile["transition_phrases"].items())[:10]:
            prompt += f"- \"{phrase}\"\n"

        prompt += "\n### Empathy Markers:\n"
        for marker, count in list(profile["empathy_markers"].items())[:10]:
            prompt += f"- \"{marker}\"\n"

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
    voice_profile = extractor.extract_voice_patterns()

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
