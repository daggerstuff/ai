#!/usr/bin/env python3
"""
Google Colab Setup Script for Edge Case Generation Pipeline
Run this first in Colab to set up the environment
"""

import importlib.util
import os
import subprocess
import sys


def install_requirements():
    """Install required packages in Colab"""
    sys.stdout.write("🔧 Installing requirements for Google Colab...\n")

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
            sys.stdout.write(f"✅ Installed {package}\n")
        except Exception as e:
            sys.stderr.write(f"❌ Failed to install {package}: {e}\n")


def setup_colab_environment():
    """Set up the Colab environment"""
    sys.stdout.write("🚀 Setting up Google Colab environment...\n")

    # Enable widgets - check if Colab modules are available
    ipython_spec = importlib.util.find_spec("IPython")
    colab_spec = importlib.util.find_spec("google.colab")

    if ipython_spec is not None and colab_spec is not None:
        # Dynamically import only if available
        ipython_module = importlib.import_module("IPython")
        colab_output = importlib.import_module("google.colab.output")

        try:
            ipython_instance = ipython_module.get_ipython()
            if ipython_instance is not None:
                colab_output.enable_custom_widget_manager()
                sys.stdout.write(
                    "✅ Enabled custom widget manager (running in Colab notebook context).\n"
                )
            else:
                sys.stdout.write(
                    "INFO: IPython environment not detected (e.g., running via '!python script.py').\n"
                )
                sys.stdout.write(
                    "   Skipping custom widget manager setup. If you use custom third-party ipywidgets,\n"
                )
                sys.stdout.write("   you may need to run the following in a separate Colab cell:\n")
                sys.stdout.write(
                    "   from google.colab import output; output.enable_custom_widget_manager()\n"
                )
        except Exception as e:
            sys.stderr.write(f"⚠️ Could not enable custom widget manager due to an error: {e}\n")
            sys.stderr.write(
                "   If you use custom third-party ipywidgets, you may need to run the following\n"
            )
            sys.stderr.write(
                "   in a separate Colab cell: from google.colab import output; output.enable_custom_widget_manager()\n"
            )
    else:
        sys.stdout.write(
            "INFO: Not running in a Google Colab environment - skipping Colab-specific widget setup.\n"
        )

    # Create output directory
    os.makedirs("colab_output", exist_ok=True)
    sys.stdout.write("✅ Created output directory\n")

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
    sys.stdout.write(instructions)


def main():
    """Main setup function"""
    sys.stdout.write("🚀 Google Colab Setup for Edge Case Generation\n")
    sys.stdout.write("=" * 60 + "\n")

    install_requirements()
    setup_colab_environment()
    display_usage_instructions()

    sys.stdout.write("\n🎉 Setup completed! Ready to generate edge cases.\n")


if __name__ == "__main__":
    main()
