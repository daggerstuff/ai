#!/usr/bin/env python3
"""
Script to update all domain references from pixelated-empathy.ai to pixelatedempathy.com
"""

import os
import re
from pathlib import Path

def update_domains_in_file(file_path):
    """Update domain references in a single file."""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Track if any changes were made
        original_content = content
        
        # Replace all variations of the old domain
        replacements = [
            ('pixelated-empathy.ai', 'pixelatedempathy.com'),
            ('api.pixelated-empathy.ai', 'api.pixelatedempathy.com'),
            ('status.pixelated-empathy.ai', 'status.pixelatedempathy.com'),
            ('api-support@pixelated-empathy.ai', 'api-support@pixelatedempathy.com'),
            ('research@pixelated-empathy.ai', 'research@pixelatedempathy.com'),
            ('billing@pixelated-empathy.ai', 'billing@pixelatedempathy.com'),
            ('emergency@pixelated-empathy.ai', 'emergency@pixelatedempathy.com'),
            ('dev-support@pixelated-empathy.ai', 'dev-support@pixelatedempathy.com'),
            ('research-support@pixelated-empathy.ai', 'research-support@pixelatedempathy.com'),
            ('research-stats@pixelated-empathy.ai', 'research-stats@pixelatedempathy.com'),
            ('data-quality@pixelated-empathy.ai', 'data-quality@pixelatedempathy.com'),
        ]
        
        for old_domain, new_domain in replacements:
            content = content.replace(old_domain, new_domain)
        
        # Write back if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Updated: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all documentation files."""
    
    # Define directories to search
    search_dirs = [
        '/home/vivi/pixelated/ai/docs',
        '/home/vivi/pixelated/ai/inference/api',
        '/home/vivi/pixelated/ai/qa/reports'
    ]
    
    # File extensions to process
    extensions = ['.md', '.py', '.js', '.json', '.yaml', '.yml', '.txt']
    
    updated_files = []
    total_files = 0
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in extensions):
                        file_path = os.path.join(root, file)
                        total_files += 1
                        
                        if update_domains_in_file(file_path):
                            updated_files.append(file_path)
    
    print(f"\n📊 Summary:")
    print(f"   Total files processed: {total_files}")
    print(f"   Files updated: {len(updated_files)}")
    
    if updated_files:
        print(f"\n📝 Updated files:")
        for file_path in updated_files:
            print(f"   - {file_path}")
    
    print(f"\n✅ Domain update completed!")

if __name__ == "__main__":
    main()
