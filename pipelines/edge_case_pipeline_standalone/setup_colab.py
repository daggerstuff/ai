#!/usr/bin/env python3
"""
Google Colab Setup Script for Edge Case Generation Pipeline
Run this first in Colab to set up the environment
"""

import os
import subprocess
import sys


def install_requirements():
    """Install required packages in Colab"""
    print("🔧 Installing requirements for Google Colab...")

    requirements = [
        "openai>=1.0.0",
        "anthropic>=0.8.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "requests>=2.31.0",
        "ipywidgets>=8.0.0",
    ]

    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except Exception as e:
            print(f"❌ Failed to install {package}: {e}")


def setup_colab_environment():
    """Set up the Colab environment"""
    print("🚀 Setting up Google Colab environment...")

    # Enable widgets
    try:
        # Attempt to import Colab-specific modules
        from google.colab import output
        import IPython

        # Check if we are running in an IPython environment (like a Colab notebook cell)
        if IPython.get_ipython() is not None:
            output.enable_custom_widget_manager()
            print("✅ Enabled custom widget manager (running in Colab notebook context).")
        else:
            print("ℹ️ IPython environment not detected (e.g., running via '!python script.py').")
            print(
                "   Skipping custom widget manager setup. If you use custom third-party ipywidgets,"
            )
            print("   you may need to run the following in a separate Colab cell:")
            print("   from google.colab import output; output.enable_custom_widget_manager()")
    except ImportError:
        # This means google.colab module is not available (e.g., running locally, not in Colab)
        print("ℹ️ Not running in a Google Colab environment - skipping Colab-specific widget setup.")
    except Exception as e:
        # Catch any other unexpected error during widget setup
        print(f"⚠️ Could not enable custom widget manager due to an error: {e}")
        print("   If you use custom third-party ipywidgets, you may need to run the following")
        print(
            "   in a separate Colab cell: from google.colab import output; output.enable_custom_widget_manager()"
        )

    # Create output directory
    os.makedirs("colab_output", exist_ok=True)
    print("✅ Created output directory")

    # The message below was originally in your main function,
    # but it's better placed after all setup steps within this function.
    # print("\n🎯 Setup complete! You can now run the edge case generator.") # Original line
    # Let main handle the final "Setup completed!" message for overall script.


def display_usage_instructions():
    """Display usage instructions for Colab"""
    instructions = """
📋 Usage Instructions for Google Colab:

1. Set your API key:
   For OpenAI: os.environ['OPENAI_API_KEY'] = 'your_key_here'
   For Anthropic: os.environ['ANTHROPIC_API_KEY'] = 'your_key_here'

2. Import the generator:
   from edge_case_generator import EdgeCaseGenerator

3. Quick test:
   exec(open('quick_start.py').read())

4. Full notebook:
   # Use Edge_Case_Generation_Pipeline.ipynb

5. Custom generation:
   generator = EdgeCaseGenerator(
       api_provider="openai",
       api_key=os.environ.get('OPENAI_API_KEY'),
       model_name="gpt-3.5-turbo",
       output_dir="colab_output"
   )
   prompts = generator.generate_prompts(scenarios_per_category=5)
   conversations = generator.generate_conversations(prompts, max_conversations=25)

💡 Tips:
- Start with small batches (5-10 conversations) to test
- Use gpt-3.5-turbo for faster/cheaper generation
- Check your API usage limits
- Download results before session expires
"""
    print(instructions)


def main():
    """Main setup function"""
    print("🚀 Google Colab Setup for Edge Case Generation")
    print("=" * 60)

    install_requirements()
    setup_colab_environment()
    display_usage_instructions()

    print("\n🎉 Setup completed! Ready to generate edge cases.")


if __name__ == "__main__":
    main()
