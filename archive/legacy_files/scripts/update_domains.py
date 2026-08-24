#!/usr/bin/env python3
"""
Script to update all domain references from pixelated-empathy.ai to pixelatedempathy.com
"""

import os


def update_domains_in_file(file_path):
    """Update domain references in a single file."""

    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Track if any changes were made
        original_content = content

        # Replace all variations of the old domain
        replacements = [
            ("pixelated-empathy.ai", "pixelatedempathy.com"),
            ("api.pixelated-empathy.ai", "api.pixelatedempathy.com"),
            ("status.pixelated-empathy.ai", "status.pixelatedempathy.com"),
            ("api-support@pixelated-empathy.ai", "api-support@pixelatedempathy.com"),
            ("research@pixelated-empathy.ai", "research@pixelatedempathy.com"),
            ("billing@pixelated-empathy.ai", "billing@pixelatedempathy.com"),
            ("emergency@pixelated-empathy.ai", "emergency@pixelatedempathy.com"),
            ("dev-support@pixelated-empathy.ai", "dev-support@pixelatedempathy.com"),
            ("research-support@pixelated-empathy.ai", "research-support@pixelatedempathy.com"),
            ("research-stats@pixelated-empathy.ai", "research-stats@pixelatedempathy.com"),
            ("data-quality@pixelated-empathy.ai", "data-quality@pixelatedempathy.com"),
        ]

        for old_domain, new_domain in replacements:
            content = content.replace(old_domain, new_domain)

        # Write back if changes were made
        if content != original_content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        return False

    except Exception:
        return False


def main():
    """Main function to update all documentation files."""

    # Define directories to search
    search_dirs = [
        "/home/vivi/pixelated/ai/docs",
        "/home/vivi/pixelated/ai/inference/api",
        "/home/vivi/pixelated/ai/infrastructure/qa/reports",
    ]

    # File extensions to process
    extensions = [".md", ".py", ".js", ".json", ".yaml", ".yml", ".txt"]

    updated_files = []
    total_files = 0

    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, _dirs, files in os.walk(search_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in extensions):
                        file_path = os.path.join(root, file)
                        total_files += 1

                        if update_domains_in_file(file_path):
                            updated_files.append(file_path)

    if updated_files:
        for file_path in updated_files:
            pass


if __name__ == "__main__":
    main()
