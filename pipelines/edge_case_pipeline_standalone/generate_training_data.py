#!/usr/bin/env python3
"""
Quick script to generate edge case training data
Run this to create the nightmare fuel edge cases for training
"""

from edge_case_generator import EdgeCaseGenerator
import sys

def main():
    print("🔥 Edge Case Training Data Generator")
    print("=" * 60)
    print("Generating challenging therapy scenarios...")
    print()
    
    # Initialize generator with Ollama (local, free)
    generator = EdgeCaseGenerator(
        api_provider="ollama",
        model_name="artifish/llama3.2-uncensored",  # Uncensored model for difficult content
        output_dir="output"
    )
    
    print("📝 Step 1: Generating prompts...")
    # Generate 20 scenarios per category = 500 total prompts
    prompts = generator.generate_prompts(scenarios_per_category=20)
    print(f"   Generated {len(prompts)} prompts across 25 categories")
    
    print("\n🤖 Step 2: Generating conversations...")
    print("   This may take a while depending on your system...")
    # Generate conversations (limit to 500 for reasonable time)
    conversations = generator.generate_conversations(
        prompts, 
        max_conversations=500
    )
    print(f"   Generated {len(conversations)} conversations")
    
    print("\n📦 Step 3: Creating training format...")
    # Convert to training format
    training_data = generator.create_training_format(conversations)
    print(f"   Created {len(training_data)} training examples")
    
    print("\n📊 Step 4: Generating report...")
    # Generate summary report
    report = generator.generate_summary_report(conversations)
    print("   Report saved to output/summary_report.md")
    
    print("\n" + "=" * 60)
    print("✅ Edge case training data generation complete!")
    print(f"📁 Output directory: output/")
    print(f"📄 Training file: output/edge_cases_training_format.jsonl")
    print(f"📊 Total training examples: {len(training_data)}")
    print()
    print("Next steps:")
    print("1. Review the generated data in output/")
    print("2. Run the integrated training pipeline to include this data")
    print("3. Train your model with the nightmare fuel edge cases!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)
