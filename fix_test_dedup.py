import re

with open("infra/cloud/distributed/test_checkpoint_system.py", "r") as f:
    content = f.read()

content = content.replace(
    "with self.manager.storage.storage.connect(",
    "import sqlite3\n            with sqlite3.connect("
)

with open("infra/cloud/distributed/test_checkpoint_system.py", "w") as f:
    f.write(content)
