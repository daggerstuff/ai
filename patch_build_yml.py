import re

with open(".github/workflows/build.yml", "r") as f:
    content = f.read()

content = content.replace("  sonarqube:\n    name: SonarQube", "  sonarqube:\n    name: SonarQube\n    if: ${{ secrets.SONAR_TOKEN != '' }}")

with open(".github/workflows/build.yml", "w") as f:
    f.write(content)
