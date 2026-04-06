import ast

with open("infrastructure/qa/enterprise_validator.py") as f:
    ast.parse(f.read())
print("Syntax OK")
