with open("core/pipelines/cli/provenance_cli.py", "r") as f:
    content = f.read()

old_code = 'console.print(\n                "[yellow]Creating minimal provenance record. Use --file for full record."\n            )'
new_code = 'console.print(\n                "[yellow]Creating minimal provenance record. "\n                "Use --file for full record."\n            )'

if old_code in content:
    content = content.replace(old_code, new_code)
    with open("core/pipelines/cli/provenance_cli.py", "w") as f:
        f.write(content)
    print("Patched long line")
else:
    print("Could not find line to replace")
