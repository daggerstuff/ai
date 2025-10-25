#!/usr/bin/env python3
"""
Quick Start Script for Edge Case Generation
Test the pipeline with a small batch
"""

from __future__ import annotations

import os
import sys

from edge_case_generator import EdgeCaseGenerator


def quick_test():
    """Run a quick test with 5 conversations"""
    print("🚀 Edge Case Generator - Quick Test", file=sys.stderr)  # noqa: T201
    print("=" * 50, file=sys.stderr)  # noqa: T201

    # Configuration - modify as needed
    api_provider = "openai"  # Change to "anthropic" or "ollama"
    api_key = os.getenv("OPENAI_API_KEY")  # or ANTHROPIC_API_KEY
    model_name = "gpt-3.5-turbo"  # or "claude-3-haiku-20240307"

    if not api_key and api_provider != "ollama":
        print("❌ Please set your API key environment variable:", file=sys.stderr)  # noqa: T201
        print(f"   export {api_provider.upper()}_API_KEY='your_key_here'", file=sys.stderr)  # noqa: T201
        return

    try:
        # Initialize generator
        print(f"🔧 Initializing generator ({api_provider})...", file=sys.stderr)  # noqa: T201
        generator = EdgeCaseGenerator(
            api_provider=api_provider,
            api_key=api_key,
            model_name=model_name,
            output_dir="quick_test_output",
        )

        # Generate small batch of prompts
        print("📝 Generating prompts (2 per category)...", file=sys.stderr)  # noqa: T201
        prompts = generator.generate_prompts(scenarios_per_category=2)

        # Generate a few conversations
        print("🤖 Generating conversations (max 10)...", file=sys.stderr)  # noqa: T201
        conversations = generator.generate_conversations(prompts, max_conversations=10)

        # Create training format
        print("🔄 Converting to training format...", file=sys.stderr)  # noqa: T201
        training_data = generator.create_training_format(conversations)

        # Generate report
        print("📄 Generating report...", file=sys.stderr)  # noqa: T201
        generator.generate_summary_report(conversations)

        # Results
        print("\n" + "=" * 50, file=sys.stderr)  # noqa: T201
        print("✅ Quick Test Results:", file=sys.stderr)  # noqa: T201
        print(f"   Generated Prompts: {len(prompts)}", file=sys.stderr)  # noqa: T201
        print(f"   Generated Conversations: {len(conversations)}", file=sys.stderr)  # noqa: T201
        print(f"   Training Examples: {len(training_data)}", file=sys.stderr)  # noqa: T201
        print("   Output Directory: quick_test_output/", file=sys.stderr)  # noqa: T201

        if conversations:
            print("\n💬 Sample Conversation:", file=sys.stderr)  # noqa: T201
            sample = conversations[0]
            print(f"   Category: {sample['category']}", file=sys.stderr)  # noqa: T201
            print(f"   Difficulty: {sample['difficulty_level']}", file=sys.stderr)  # noqa: T201
            if sample.get("qa_pairs"):
                qa = sample["qa_pairs"][0]
                print(f"   Therapist: {qa['prompt'][:60]}...", file=sys.stderr)  # noqa: T201
                print(f"   Client: {qa['response'][:60]}...", file=sys.stderr)  # noqa: T201

        print("\n🎉 Quick test completed successfully!", file=sys.stderr)  # noqa: T201
        print("📁 Check the 'quick_test_output' directory for results", file=sys.stderr)  # noqa: T201

    except Exception as e:
        print(f"❌ Error during quick test: {e}", file=sys.stderr)  # noqa: T201
        print("\nTroubleshooting:", file=sys.stderr)  # noqa: T201
        print("1. Check your API key is correct", file=sys.stderr)  # noqa: T201
        print("2. Verify internet connection", file=sys.stderr)  # noqa: T201
        print("3. Check API provider spelling", file=sys.stderr)  # noqa: T201
        print("4. For Ollama, ensure it's running: ollama serve", file=sys.stderr)  # noqa: T201


if __name__ == "__main__":
    quick_test()
