#!/usr/bin/env python3
"""
Prompt Generation System for LoRA Training Data Conversion
Converts therapeutic segments into authentic question-answer training pairs
"""

import json
import re
import random
from typing import Dict, List, Tuple
from pathlib import Path

class TherapeuticPromptGenerator:
    def __init__(self):
        # Style-specific prompt templates
        self.prompt_templates = {
            "therapeutic": [
                "I'm struggling with {topic}. Can you help me understand what's happening?",
                "I've been dealing with {topic} and I don't know how to heal from it. What should I know?",
                "Can you explain how {topic} affects someone and what the path to recovery looks like?",
                "I'm working through {topic} in therapy. What insights can you share about this?",
                "How does {topic} impact a person's life, and what are the steps to healing?",
                "I'm trying to understand {topic} better. What would you want someone to know about this?",
                "What should someone know about {topic} and how it affects their relationships and healing?",
                "I'm dealing with {topic} and feeling stuck. What perspective can you offer?",
            ],
            "educational": [
                "Can you explain what {topic} is and how it works?",
                "I want to understand {topic} better. Can you break it down for me?",
                "What should I know about {topic} from a clinical perspective?",
                "How does {topic} develop and what are the key things to understand about it?",
                "Can you teach me about {topic} and its impact on mental health?",
                "What are the important facts about {topic} that people should understand?",
                "I'm learning about {topic}. What are the key concepts I should grasp?",
                "From a therapeutic standpoint, how would you explain {topic}?",
            ],
            "empathetic": [
                "I'm really struggling with {topic} and feeling alone. Can you help?",
                "I feel so overwhelmed by {topic}. I need someone to understand what I'm going through.",
                "I'm in pain because of {topic}. Can you help me feel less alone?",
                "I don't know how to cope with {topic} anymore. I need support and understanding.",
                "I'm hurting from {topic} and I need to know that someone gets it.",
                "I feel broken because of {topic}. Can you help me see that I'm not alone?",
                "I'm struggling to make sense of {topic} and I need compassionate guidance.",
                "I feel lost dealing with {topic}. Can you offer me hope and understanding?",
            ],
            "practical": [
                "What specific steps can I take to deal with {topic}?",
                "I need practical advice for handling {topic}. What should I do?",
                "What are concrete strategies for managing {topic} in daily life?",
                "Can you give me actionable steps to work through {topic}?",
                "What practical tools or techniques help with {topic}?",
                "I need a clear plan for addressing {topic}. What do you recommend?",
                "What are the most effective approaches for dealing with {topic}?",
                "Can you outline practical steps someone can take to heal from {topic}?",
            ]
        }
        
        # Topic extraction patterns
        self.topic_patterns = [
            r"trauma|PTSD|complex trauma|betrayal trauma|childhood trauma",
            r"narcissist|narcissism|emotional abuse|manipulation|gaslighting",
            r"anxiety|depression|emotional dysregulation|panic|fear",
            r"attachment|relationships|trust|intimacy|boundaries",
            r"shame|guilt|self-worth|identity|self-esteem",
            r"healing|recovery|therapy|treatment|growth",
            r"family|parents|childhood|scapegoat|dysfunction",
            r"codependency|people pleasing|validation|approval",
            r"anger|rage|emotional regulation|triggers",
            r"stress|burnout|overwhelm|exhaustion"
        ]

    def extract_key_topics(self, text: str) -> List[str]:
        """Extract key therapeutic topics from segment text"""
        topics = []
        text_lower = text.lower()
        
        # Extract specific topics using patterns
        for pattern in self.topic_patterns:
            matches = re.findall(pattern, text_lower)
            topics.extend(matches)
        
        # Extract noun phrases that might be topics
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences[:3]:  # Focus on first few sentences
            words = sentence.split()
            if len(words) > 5:
                # Look for key therapeutic terms
                for word in words:
                    if any(term in word.lower() for term in ['trauma', 'heal', 'therap', 'relation', 'emotion']):
                        context = ' '.join(words[max(0, words.index(word)-2):words.index(word)+3])
                        topics.append(context.strip())
                        break
        
        # Fallback topics if none found
        if not topics:
            topics = ["emotional healing", "personal growth", "mental health"]
        
        return list(set(topics))[:3]  # Return up to 3 unique topics

    def generate_prompt(self, segment: Dict) -> str:
        """Generate an authentic therapeutic question for a segment"""
        style = segment['style']
        text = segment['text']
        
        # Extract topics from the segment
        topics = self.extract_key_topics(text)
        selected_topic = random.choice(topics) if topics else "healing"
        
        # Clean up topic for insertion
        topic = selected_topic.lower().strip()
        
        # Select appropriate template
        templates = self.prompt_templates.get(style, self.prompt_templates['therapeutic'])
        template = random.choice(templates)
        
        # Generate the prompt
        prompt = template.format(topic=topic)
        
        return prompt

    def create_training_pair(self, segment: Dict) -> Dict:
        """Convert a segment into a training pair"""
        prompt = self.generate_prompt(segment)
        
        return {
            "input": prompt,
            "output": segment['text'],
            "style": segment['style'],
            "confidence": segment['confidence'],
            "quality": segment['quality'],
            "source": segment['source'],
            "file": segment['file']
        }

    def process_segments_file(self, input_path: Path, output_path: Path) -> Dict:
        """Process a segments file and convert to training pairs"""
        with open(input_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        training_pairs = []
        stats = {"total": len(segments), "processed": 0, "errors": 0}
        
        for segment in segments:
            try:
                training_pair = self.create_training_pair(segment)
                training_pairs.append(training_pair)
                stats["processed"] += 1
            except Exception as e:
                print(f"Error processing segment: {e}")
                stats["errors"] += 1
        
        # Save training pairs
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(training_pairs, f, indent=2, ensure_ascii=False)
        
        return stats

def main():
    """Convert all segment files to training pairs"""
    generator = TherapeuticPromptGenerator()
    
    segments_dir = Path("/root/pixelated/ai/data/training_segments")
    output_dir = Path("/root/pixelated/ai/data/lora_training")
    output_dir.mkdir(exist_ok=True)
    
    total_stats = {"total": 0, "processed": 0, "errors": 0}
    
    # Process each segment file
    for segment_file in segments_dir.glob("*.json"):
        if segment_file.name == "enhanced_summary.json":
            continue
            
        print(f"Processing {segment_file.name}...")
        
        output_file = output_dir / f"training_{segment_file.name}"
        stats = generator.process_segments_file(segment_file, output_file)
        
        print(f"  Processed: {stats['processed']}/{stats['total']} segments")
        if stats['errors'] > 0:
            print(f"  Errors: {stats['errors']}")
        
        # Update totals
        for key in total_stats:
            total_stats[key] += stats[key]
    
    print(f"\nTotal conversion complete:")
    print(f"  Processed: {total_stats['processed']}/{total_stats['total']} segments")
    print(f"  Success rate: {total_stats['processed']/total_stats['total']*100:.1f}%")
    print(f"  Output directory: {output_dir}")

if __name__ == "__main__":
    main()
