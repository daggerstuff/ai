#!/usr/bin/env python3
"""
Quick Start Script for Edge Case Generation
Test the pipeline with a small batch
"""

import os
from edge_case_generator import EdgeCaseGenerator


def quick_test():
    """Run a quick test with 5 conversations"""

    print("🚀 Edge Case Generator - Quick Test")
    print("=" * 50)

    # Configuration - modify as needed
    API_PROVIDER = "openai"  # Change to "anthropic" or "ollama"
    API_KEY = os.getenv("OPENAI_API_KEY")  # or ANTHROPIC_API_KEY
    MODEL_NAME = "gpt-3.5-turbo"  # or "claude-3-haiku-20240307"

    if not API_KEY and API_PROVIDER != "ollama":
        print(f"❌ Please set your API key environment variable:")
        print(f"   export {API_PROVIDER.upper()}_API_KEY='your_key_here'")
        return

    try:
        # Initialize generator
        print(f"🔧 Initializing generator ({API_PROVIDER})...")
        generator = EdgeCaseGenerator(
            api_provider=API_PROVIDER,
            api_key=API_KEY,
            model_name=MODEL_NAME,
            output_dir="quick_test_output",
        )

        # Generate small batch of prompts
        print("📝 Generating prompts (2 per category)...")
        prompts = generator.generate_prompts(scenarios_per_category=2)

        # Generate a few conversations
        print("🤖 Generating conversations (max 10)...")
        conversations = generator.generate_conversations(prompts, max_conversations=10)

        # Create training format
        print("🔄 Converting to training format...")
        training_data = generator.create_training_format(conversations)

        # Generate report
        print("📄 Generating report...")
        report = generator.generate_summary_report(conversations)

        # Results
        print("\n" + "=" * 50)
        print("✅ Quick Test Results:")
        print(f"   Generated Prompts: {len(prompts)}")
        print(f"   Generated Conversations: {len(conversations)}")
        print(f"   Training Examples: {len(training_data)}")
        print(f"   Output Directory: quick_test_output/")

        if conversations:
            print(f"\n💬 Sample Conversation:")
            sample = conversations[0]
            print(f"   Category: {sample['category']}")
            print(f"   Difficulty: {sample['difficulty_level']}")
            if sample.get("qa_pairs"):
                qa = sample["qa_pairs"][0]
                print(f"   Therapist: {qa['prompt'][:60]}...")
                print(f"   Client: {qa['response'][:60]}...")

        print("\n🎉 Quick test completed successfully!")
        print("📁 Check the 'quick_test_output' directory for results")

    except Exception as e:
        print(f"❌ Error during quick test: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API key is correct")
        print("2. Verify internet connection")
        print("3. Check API provider spelling")
        print("4. For Ollama, ensure it's running: ollama serve")


if __name__ == "__main__":
    quick_test()
