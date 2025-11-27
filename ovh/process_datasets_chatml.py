#!/usr/bin/env python3
"""
Comprehensive Dataset Processing Pipeline for ChatML Format
Pixelated Empathy - Wayfarer-2-12B Fine-Tuning

This script processes all consolidated datasets into proper ChatML format
for staged training (Foundation → Reasoning → Voice).

Target Format (ChatML/ShareGPT):
{
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}

Dataset Categories:
1. Professional (Foundation): Natural therapeutic dialogue patterns
2. CoT Reasoning: Clinical reasoning with explicit thought chains  
3. Priority: Curated high-quality therapeutic conversations
4. Reddit: Mental health community conversations (supplementary)

Author: Pixelated Empathy AI Team
"""

import json
import csv
import os
import re
import sys
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Any
from collections import defaultdict
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dataset_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# System prompts for different training stages
SYSTEM_PROMPTS = {
    "foundation": """You are a compassionate and skilled mental health counselor. 
Your role is to provide supportive, empathetic responses while maintaining professional boundaries. 
Listen actively, validate emotions, and help clients explore their feelings and develop coping strategies.""",
    
    "reasoning": """You are a clinical mental health expert who thinks through problems systematically.
When responding, consider the psychological context, potential underlying issues, and evidence-based approaches.
Demonstrate your clinical reasoning process while remaining warm and supportive.""",
    
    "voice": """You are a therapeutic guide in the style of Tim Fletcher, combining trauma-informed insights 
with accessible, educational explanations. Use clear analogies, validate experiences, 
and help people understand the 'why' behind their patterns while offering practical pathways forward."""
}

class DatasetProcessor:
    """Main processor class for converting various dataset formats to ChatML."""
    
    def __init__(self, base_dir: str = "~/datasets/consolidated"):
        self.base_dir = Path(base_dir).expanduser()
        self.output_dir = self.base_dir / "chatml_formatted"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = defaultdict(lambda: {"total": 0, "valid": 0, "skipped": 0, "tokens_approx": 0})
        self.seen_hashes = set()
        self.all_conversations = []
        
    def hash_conversation(self, messages: list[dict]) -> str:
        """Create hash for deduplication."""
        content = json.dumps(messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_duplicate(self, messages: list[dict]) -> bool:
        """Check if conversation is a duplicate."""
        h = self.hash_conversation(messages)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (words * 1.3)."""
        return int(len(text.split()) * 1.3)
    
    def validate_conversation(self, messages: list[dict], min_turns: int = 2) -> bool:
        """Validate a conversation has minimum content."""
        if len(messages) < min_turns:
            return False
        
        # Check for actual content
        user_content = sum(1 for m in messages if m.get("role") == "user" and len(m.get("content", "").strip()) > 10)
        assistant_content = sum(1 for m in messages if m.get("role") == "assistant" and len(m.get("content", "").strip()) > 10)
        
        return user_content >= 1 and assistant_content >= 1
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not isinstance(text, str):
            return ""
        
        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()
        
        return text
    
    # =========== PROFESSIONAL DATASET PROCESSORS ===========
    
    def process_mental_health_counseling(self) -> list[dict]:
        """Process mental_health_counseling_conversations (Context/Response format)."""
        logger.info("Processing: mental_health_counseling_conversations")
        conversations = []
        
        data_file = self.base_dir / "professional/mental_health_counseling_conversations/combined_dataset.json"
        if not data_file.exists():
            logger.warning(f"File not found: {data_file}")
            return conversations
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Handle JSONL or JSON array
                if content.strip().startswith('['):
                    data = json.loads(content)
                else:
                    data = [json.loads(line) for line in content.strip().split('\n') if line.strip()]
            
            for item in data:
                context = self.clean_text(item.get("Context", ""))
                response = self.clean_text(item.get("Response", ""))
                
                if not context or not response:
                    self.stats["mental_health_counseling"]["skipped"] += 1
                    continue
                
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPTS["foundation"]},
                    {"role": "user", "content": context},
                    {"role": "assistant", "content": response}
                ]
                
                if not self.is_duplicate(messages):
                    conversations.append({"messages": messages, "source": "mental_health_counseling"})
                    self.stats["mental_health_counseling"]["valid"] += 1
                    self.stats["mental_health_counseling"]["tokens_approx"] += self.estimate_tokens(context + response)
                else:
                    self.stats["mental_health_counseling"]["skipped"] += 1
                
                self.stats["mental_health_counseling"]["total"] += 1
                
        except Exception as e:
            logger.error(f"Error processing mental_health_counseling: {e}")
        
        return conversations
    
    def process_therapist_sft(self) -> list[dict]:
        """Process therapist-sft-format (CSV with human/gpt turns)."""
        logger.info("Processing: therapist-sft-format")
        conversations = []
        
        data_file = self.base_dir / "professional/therapist-sft-format/train.csv"
        if not data_file.exists():
            logger.warning(f"File not found: {data_file}")
            return conversations
        
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    text = row.get("text", "")
                    if not text:
                        continue
                    
                    # Parse human/gpt turns
                    messages = [{"role": "system", "content": SYSTEM_PROMPTS["foundation"]}]
                    
                    # Split by 'human:' and 'gpt:' markers
                    pattern = r'(human:|gpt:)'
                    parts = re.split(pattern, text, flags=re.IGNORECASE)
                    
                    current_role = None
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        if part.lower() == "human:":
                            current_role = "user"
                        elif part.lower() == "gpt:":
                            current_role = "assistant"
                        elif current_role and part:
                            content = self.clean_text(part)
                            if content:
                                messages.append({"role": current_role, "content": content})
                    
                    if self.validate_conversation(messages) and not self.is_duplicate(messages):
                        conversations.append({"messages": messages, "source": "therapist_sft"})
                        self.stats["therapist_sft"]["valid"] += 1
                        self.stats["therapist_sft"]["tokens_approx"] += self.estimate_tokens(text)
                    else:
                        self.stats["therapist_sft"]["skipped"] += 1
                    
                    self.stats["therapist_sft"]["total"] += 1
                    
        except Exception as e:
            logger.error(f"Error processing therapist_sft: {e}")
        
        return conversations
    
    def process_soulchat(self) -> list[dict]:
        """Process SoulChat2.0 (ShareGPT format already)."""
        logger.info("Processing: SoulChat2.0")
        conversations = []
        
        soulchat_dir = self.base_dir / "professional/SoulChat2.0/PsyDTCorpus"
        if not soulchat_dir.exists():
            logger.warning(f"Directory not found: {soulchat_dir}")
            return conversations
        
        for json_file in soulchat_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        messages_raw = item.get("messages", item.get("conversations", []))
                        if not messages_raw:
                            continue
                        
                        # Convert to standard format
                        messages = [{"role": "system", "content": SYSTEM_PROMPTS["foundation"]}]
                        for msg in messages_raw:
                            role = msg.get("role", "").lower()
                            content = self.clean_text(msg.get("content", ""))
                            
                            if role in ["user", "human"]:
                                role = "user"
                            elif role in ["assistant", "gpt", "bot"]:
                                role = "assistant"
                            else:
                                continue
                            
                            if content:
                                messages.append({"role": role, "content": content})
                        
                        if self.validate_conversation(messages) and not self.is_duplicate(messages):
                            conversations.append({"messages": messages, "source": "soulchat"})
                            self.stats["soulchat"]["valid"] += 1
                        else:
                            self.stats["soulchat"]["skipped"] += 1
                        
                        self.stats["soulchat"]["total"] += 1
                        
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        return conversations
    
    def process_counsel_chat(self) -> list[dict]:
        """Process counsel-chat dataset."""
        logger.info("Processing: counsel-chat")
        conversations = []
        
        counsel_dir = self.base_dir / "professional/counsel-chat"
        if not counsel_dir.exists():
            logger.warning(f"Directory not found: {counsel_dir}")
            return conversations
        
        for json_file in counsel_dir.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                items = data if isinstance(data, list) else [data]
                for item in items:
                    # Handle various formats
                    question = self.clean_text(item.get("questionText", item.get("question", item.get("Context", ""))))
                    answer = self.clean_text(item.get("answerText", item.get("answer", item.get("Response", ""))))
                    
                    if not question or not answer:
                        continue
                    
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPTS["foundation"]},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                    
                    if not self.is_duplicate(messages):
                        conversations.append({"messages": messages, "source": "counsel_chat"})
                        self.stats["counsel_chat"]["valid"] += 1
                    else:
                        self.stats["counsel_chat"]["skipped"] += 1
                    
                    self.stats["counsel_chat"]["total"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        return conversations
    
    def process_psych8k(self) -> list[dict]:
        """Process Psych8k dataset."""
        logger.info("Processing: Psych8k")
        conversations = []
        
        psych_dir = self.base_dir / "professional/Psych8k"
        if not psych_dir.exists():
            return conversations
        
        for json_file in psych_dir.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                items = data if isinstance(data, list) else [data]
                for item in items:
                    # Similar parsing to counsel_chat
                    prompt = self.clean_text(item.get("prompt", item.get("input", "")))
                    response = self.clean_text(item.get("response", item.get("output", "")))
                    
                    if not prompt or not response:
                        continue
                    
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPTS["foundation"]},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response}
                    ]
                    
                    if not self.is_duplicate(messages):
                        conversations.append({"messages": messages, "source": "psych8k"})
                        self.stats["psych8k"]["valid"] += 1
                    else:
                        self.stats["psych8k"]["skipped"] += 1
                    
                    self.stats["psych8k"]["total"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        return conversations
    
    # =========== COT REASONING PROCESSORS ===========
    
    def process_cot_dataset(self, dataset_name: str) -> list[dict]:
        """Process a Chain-of-Thought reasoning dataset."""
        logger.info(f"Processing COT: {dataset_name}")
        conversations = []
        
        dataset_dir = self.base_dir / "cot_reasoning" / dataset_name
        if not dataset_dir.exists():
            logger.warning(f"Directory not found: {dataset_dir}")
            return conversations
        
        # Find JSON files
        json_files = list(dataset_dir.glob("*.json"))
        if not json_files:
            json_files = list(dataset_dir.rglob("*.json"))
        
        for json_file in json_files:
            if json_file.name.startswith('.') or 'readme' in json_file.name.lower():
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    # CoT format has 'answer' with reasoning in metadata
                    answer = self.clean_text(item.get("answer", ""))
                    metadata = item.get("metadata", {})
                    reasoning = self.clean_text(metadata.get("reasoning", ""))
                    
                    if not answer:
                        self.stats[f"cot_{dataset_name}"]["skipped"] += 1
                        continue
                    
                    # For CoT, we create a synthetic question from the context
                    # and include the reasoning as part of the response
                    question = item.get("question", item.get("prompt", ""))
                    if not question:
                        # Try to infer a question from the answer context
                        question = "Can you help me understand this situation and provide guidance?"
                    question = self.clean_text(question)
                    
                    # Build response with reasoning (if available)
                    if reasoning:
                        response = f"Let me think through this carefully.\n\n**Reasoning:**\n{reasoning}\n\n**Response:**\n{answer}"
                    else:
                        response = answer
                    
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPTS["reasoning"]},
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": response}
                    ]
                    
                    if self.validate_conversation(messages, min_turns=2) and not self.is_duplicate(messages):
                        conversations.append({
                            "messages": messages, 
                            "source": f"cot_{dataset_name}",
                            "cot_type": dataset_name
                        })
                        self.stats[f"cot_{dataset_name}"]["valid"] += 1
                        self.stats[f"cot_{dataset_name}"]["tokens_approx"] += self.estimate_tokens(response)
                    else:
                        self.stats[f"cot_{dataset_name}"]["skipped"] += 1
                    
                    self.stats[f"cot_{dataset_name}"]["total"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        return conversations
    
    def process_all_cot_datasets(self) -> list[dict]:
        """Process all CoT reasoning datasets."""
        logger.info("Processing all CoT reasoning datasets...")
        all_cot = []
        
        cot_dir = self.base_dir / "cot_reasoning"
        if not cot_dir.exists():
            logger.warning("CoT reasoning directory not found")
            return all_cot
        
        for dataset_dir in cot_dir.iterdir():
            if dataset_dir.is_dir() and not dataset_dir.name.startswith('.'):
                cot_data = self.process_cot_dataset(dataset_dir.name)
                all_cot.extend(cot_data)
                logger.info(f"  {dataset_dir.name}: {len(cot_data)} conversations")
        
        return all_cot
    
    # =========== PROCESSED DATA HANDLERS ===========
    
    def process_already_processed(self) -> list[dict]:
        """Process datasets that were already processed in earlier pipeline runs."""
        logger.info("Processing: already processed datasets")
        conversations = []
        
        processed_dir = self.base_dir / "processed"
        if not processed_dir.exists():
            return conversations
        
        # Look for JSONL files that are already in conversation format
        for jsonl_file in processed_dir.rglob("*.jsonl"):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        
                        item = json.loads(line)
                        
                        # Check if already in messages format
                        if "messages" in item:
                            messages = item["messages"]
                        elif "conversation" in item:
                            messages = item["conversation"]
                        elif "prompt" in item and "response" in item:
                            messages = [
                                {"role": "system", "content": SYSTEM_PROMPTS["foundation"]},
                                {"role": "user", "content": self.clean_text(item["prompt"])},
                                {"role": "assistant", "content": self.clean_text(item["response"])}
                            ]
                        else:
                            continue
                        
                        # Ensure system prompt exists
                        if messages and messages[0].get("role") != "system":
                            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPTS["foundation"]})
                        
                        if self.validate_conversation(messages) and not self.is_duplicate(messages):
                            conversations.append({
                                "messages": messages,
                                "source": f"processed_{jsonl_file.parent.name}"
                            })
                            self.stats["processed"]["valid"] += 1
                        else:
                            self.stats["processed"]["skipped"] += 1
                        
                        self.stats["processed"]["total"] += 1
                        
            except Exception as e:
                logger.error(f"Error processing {jsonl_file}: {e}")
        
        return conversations
    
    # =========== MAIN PIPELINE ===========
    
    def run_full_pipeline(self):
        """Run the complete data processing pipeline."""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE DATASET PROCESSING PIPELINE")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Stage 1: Foundation datasets (Professional therapeutic)
        logger.info("\n📚 STAGE 1: Processing Foundation Datasets")
        foundation_data = []
        foundation_data.extend(self.process_mental_health_counseling())
        foundation_data.extend(self.process_therapist_sft())
        foundation_data.extend(self.process_soulchat())
        foundation_data.extend(self.process_counsel_chat())
        foundation_data.extend(self.process_psych8k())
        
        # Stage 2: Reasoning datasets (CoT)
        logger.info("\n🧠 STAGE 2: Processing Reasoning Datasets")
        reasoning_data = self.process_all_cot_datasets()
        
        # Stage 3: Already processed data (supplementary)
        logger.info("\n📁 STAGE 3: Processing Already Processed Data")
        processed_data = self.process_already_processed()
        
        # Combine and create splits
        logger.info("\n🔀 Creating Training Splits")
        
        # Shuffle within each category
        random.seed(42)
        random.shuffle(foundation_data)
        random.shuffle(reasoning_data)
        random.shuffle(processed_data)
        
        # Save staged training data
        self.save_staged_data("stage1_foundation", foundation_data)
        self.save_staged_data("stage2_reasoning", reasoning_data)
        self.save_staged_data("stage3_supplementary", processed_data)
        
        # Create combined dataset with proper proportions for mixed training option
        combined = []
        combined.extend(foundation_data)
        combined.extend(reasoning_data)
        random.shuffle(combined)
        
        # Create train/val split (99/1)
        split_idx = int(len(combined) * 0.99)
        train_data = combined[:split_idx]
        val_data = combined[split_idx:]
        
        self.save_jsonl(train_data, self.output_dir / "train.jsonl")
        self.save_jsonl(val_data, self.output_dir / "val.jsonl")
        
        # Generate statistics report
        self.generate_report(start_time)
        
        logger.info("\n✅ PIPELINE COMPLETE!")
        logger.info(f"Output directory: {self.output_dir}")
    
    def save_staged_data(self, stage_name: str, data: list[dict]):
        """Save data for a specific training stage."""
        if not data:
            logger.warning(f"No data for {stage_name}")
            return
        
        # Split train/val
        split_idx = int(len(data) * 0.99)
        train = data[:split_idx]
        val = data[split_idx:]
        
        stage_dir = self.output_dir / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_jsonl(train, stage_dir / "train.jsonl")
        self.save_jsonl(val, stage_dir / "val.jsonl")
        
        logger.info(f"  {stage_name}: {len(train)} train, {len(val)} val")
    
    def save_jsonl(self, data: list[dict], path: Path):
        """Save data as JSONL."""
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def generate_report(self, start_time: datetime):
        """Generate processing statistics report."""
        end_time = datetime.now()
        duration = end_time - start_time
        
        report = {
            "timestamp": end_time.isoformat(),
            "duration_seconds": duration.total_seconds(),
            "output_directory": str(self.output_dir),
            "datasets": dict(self.stats),
            "summary": {
                "total_conversations": sum(s["valid"] for s in self.stats.values()),
                "total_skipped": sum(s["skipped"] for s in self.stats.values()),
                "estimated_tokens": sum(s["tokens_approx"] for s in self.stats.values()),
                "unique_hashes": len(self.seen_hashes)
            }
        }
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        
        for name, stats in sorted(self.stats.items()):
            logger.info(f"{name}:")
            logger.info(f"  Total: {stats['total']}, Valid: {stats['valid']}, Skipped: {stats['skipped']}")
        
        logger.info("-" * 60)
        logger.info(f"TOTAL VALID CONVERSATIONS: {report['summary']['total_conversations']:,}")
        logger.info(f"ESTIMATED TOKENS: {report['summary']['estimated_tokens']:,}")
        logger.info(f"UNIQUE HASHES: {report['summary']['unique_hashes']:,}")
        logger.info(f"DURATION: {duration}")
        
        # Save report
        report_path = self.output_dir / "processing_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nReport saved to: {report_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process datasets for ChatML training format")
    parser.add_argument("--base-dir", default="~/datasets/consolidated", 
                       help="Base directory for consolidated datasets")
    parser.add_argument("--stage", choices=["all", "foundation", "reasoning", "processed"],
                       default="all", help="Processing stage to run")
    
    args = parser.parse_args()
    
    processor = DatasetProcessor(base_dir=args.base_dir)
    
    if args.stage == "all":
        processor.run_full_pipeline()
    elif args.stage == "foundation":
        data = []
        data.extend(processor.process_mental_health_counseling())
        data.extend(processor.process_therapist_sft())
        data.extend(processor.process_soulchat())
        processor.save_staged_data("foundation_only", data)
    elif args.stage == "reasoning":
        data = processor.process_all_cot_datasets()
        processor.save_staged_data("reasoning_only", data)
    elif args.stage == "processed":
        data = processor.process_already_processed()
        processor.save_staged_data("processed_only", data)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()

