#!/usr/bin/env python3
"""
Package Edge Case Generation Pipeline for Transfer
Creates a zip file ready for Google Drive/Colab
"""

import zipfile
import os
from pathlib import Path
from datetime import datetime


def create_transfer_package():
    """Create a comprehensive package for transfer"""

    # Get current directory
    current_dir = Path(__file__).parent

    # Define files to include
    files_to_include = [
        "edge_case_generator.py",
        "requirements.txt",
        "README.md",
        "quick_start.py",
        "setup_colab.py",
        "config_example.py",
        "Edge_Case_Generation_Pipeline.ipynb",
    ]

    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"pixelated_empathy_edge_case_pipeline_{timestamp}.zip"

    print(f"📦 Creating transfer package: {zip_filename}")
    print("=" * 50)

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file_name in files_to_include:
            file_path = current_dir / file_name
            if file_path.exists():
                zipf.write(file_path, file_name)
                print(f"✅ Added: {file_name}")
            else:
                print(f"⚠️ Missing: {file_name}")

        # Add a quick start guide
        quickstart_content = create_quickstart_guide()
        zipf.writestr("QUICKSTART.md", quickstart_content)
        print("✅ Added: QUICKSTART.md")

    # Get zip file size
    zip_size = os.path.getsize(zip_filename)
    print("\n" + "=" * 50)
    print(f"📦 Package created: {zip_filename}")
    print(f"💾 Size: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")
    print(f"📁 Current directory: {os.getcwd()}")

    print(f"\n🚀 Ready for transfer! Upload {zip_filename} to Google Drive.")

    return zip_filename


def create_quickstart_guide():
    """Create a quick start guide for the package"""
    return """# Pixelated Empathy Edge Case Pipeline - Quick Start

## 🚀 Getting Started in Google Colab

1. **Upload and Extract**
   - Upload this zip file to Google Drive
   - In Colab: File → Open notebook → Upload and extract files

2. **Setup Environment** 
   ```python
   # Run this first in Colab
   exec(open('setup_colab.py').read())
   ```

3. **Set API Key**
   ```python
   import os
   # For OpenAI
   os.environ['OPENAI_API_KEY'] = 'your_key_here'
   # For Anthropic  
   os.environ['ANTHROPIC_API_KEY'] = 'your_key_here'
   ```

4. **Quick Test**
   ```python
   exec(open('quick_start.py').read())
   ```

5. **Full Pipeline**
   - Open `Edge_Case_Generation_Pipeline.ipynb`
   - Run cells step by step

## 📋 File Overview

- `edge_case_generator.py` - Main generator class
- `Edge_Case_Generation_Pipeline.ipynb` - Interactive notebook
- `quick_start.py` - Quick test script
- `setup_colab.py` - Colab environment setup
- `config_example.py` - Configuration template
- `requirements.txt` - Dependencies
- `README.md` - Full documentation

## 💡 Pro Tips

- Start with 5-10 conversations for testing
- Use gpt-3.5-turbo for speed and cost efficiency
- Download results before Colab session expires
- Check API rate limits and usage

## 🎯 Expected Results

- 25 edge case categories
- Professional therapy dialogues
- Training-ready format
- Comprehensive reports

Happy generating! 🤖✨
"""


if __name__ == "__main__":
    create_transfer_package()
