"""Risk Stratification Service — FastAPI service for clinical risk assessment.

Accepts PHQ-9, GAD-7, and C-SSRS scores plus clinical note context,
returns a risk level (low/medium/high/crisis) with recommended actions.

Architecture mirrors the note_drafting service pattern:
  config.py  — pydantic settings (NIM endpoint, BAA flag, retries)
  models.py  — request/response schemas, RiskLevel enum
  scoring.py — deterministic scoring for PHQ-9, GAD-7, C-SSRS
  phi.py     — PHI redaction helpers for safe logging
  service.py — orchestration: NIM call with fallback to mock
  main.py    — FastAPI app, dependency injection, endpoints
"""
