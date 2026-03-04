
def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove unused imports
    content = content.replace("from pathlib import Path\n", "")
    content = content.replace("from datetime import datetime\n", "")
    content = content.replace("import json\n", "import json\n", 1) # Keep one if needed, but ruff said unused

    if "process_60gb" in filepath:
        content = content.replace("import json\n", "")
        content = content.replace("import os\n", "")

    # Use raw strings for templates to avoid SyntaxWarning
    content = content.replace("Template('''", "Template(r'''")

    # Adjust backslashes in raw strings: \\\\ becomes \\
    # In the previous version I used \\\\ which in normal string is \\
    # In raw string, \\\\ stays \\\\ which might be too many.
    # The shell script needs \b. In a raw string r'\b' is \b.
    # So if I want \b in the file, I use r'\b'.
    # If I want \\b in the file, I use r'\\b'.

    if "Template(r'''" in content:
        # If it was '''\\\\b''' (which is \\b in memory), and I change to r'''\\\\b''', it becomes \\\\b in memory.
        # We want \\b in memory so it writes \b to the file (for the python -c command).
        # Actually, the python -c command in the shell script uses r'\b'.
        # So we want r'\b' to appear in the shell script.
        # To get r'\b' in the shell script, we need the string in Python to contain \b.
        # In a raw string r'\b' is \b.

        # Let's just manually fix the specific lines.
        content = content.replace(r'\\\\b', r'\\b')
        content = content.replace(r'\\\\d', r'\\d')
        content = content.replace(r'\\\\.', r'\\.')
        content = content.replace(r'\\\\+', r'\\+')
        content = content.replace(r'\\`', '`') # Fix the escaped backtick

    with open(filepath, 'w') as f:
        f.write(content)

fix_file('training/scripts/ovh_60gb_final.py')
fix_file('training/scripts/process_60gb_ovh_final.py')
