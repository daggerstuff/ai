import re

with open("infra/cloud/distributed/test_checkpoint_system.py", "r") as f:
    content = f.read()

# Replace naive datetimes with aware ones
content = content.replace("datetime.utcnow()", "datetime.now(timezone.utc)")

# Add timezone import if missing
if "from datetime import datetime, timezone" not in content:
    content = content.replace("from datetime import datetime", "from datetime import datetime, timezone")

with open("infra/cloud/distributed/test_checkpoint_system.py", "w") as f:
    f.write(content)
