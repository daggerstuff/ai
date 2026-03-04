import re

with open("infra/cloud/distributed/checkpoint_system.py", "r") as f:
    content = f.read()

# Fix the line that goes beyond 88 chars that we introduced or are easy to fix
content = content.replace(
    "json.dumps(asdict(metadata), default=lambda o: o.value if hasattr(o, 'value') else (o.isoformat() if hasattr(o, 'isoformat') else str(o)))",
    "json.dumps(\n                        asdict(metadata),\n                        default=lambda o: (\n                            o.value\n                            if hasattr(o, \"value\")\n                            else (o.isoformat() if hasattr(o, \"isoformat\") else str(o))\n                        ),\n                    )"
)

with open("infra/cloud/distributed/checkpoint_system.py", "w") as f:
    f.write(content)
