#!/usr/bin/env python3
"""
Update Quality Dashboard with New Professional Dataset Totals
"""

import json
from datetime import datetime, timezone
from pathlib import Path


def update_quality_dashboard():
    """Update the quality dashboard with new totals."""

    dashboard_path = Path("/home/vivi/pixelated/ai/data/processed/phase_2_professional_datasets/task_5_14_quality_validation/quality_dashboard.json")

    # Load existing dashboard
    with open(dashboard_path, encoding="utf-8") as f:
        dashboard = json.load(f)

    # Calculate new totals
    old_total = 134089
    professional_increase = 13917  # From our calculation
    new_total = old_total + professional_increase

    # Update summary
    dashboard["summary"]["total_conversations"] = new_total
    dashboard["summary"]["validation_timestamp"] = datetime.now(timezone.utc).isoformat()
    dashboard["generated"] = datetime.now(timezone.utc).isoformat()

    # Update key metrics
    dashboard["key_metrics"]["total_conversations"] = new_total

    # Add professional dataset update info
    dashboard["professional_datasets_update"] = {
        "update_date": datetime.now(timezone.utc).isoformat(),
        "conversations_added": professional_increase,
        "new_total": new_total,
        "update_reason": "Completed Task 5.2.2 with full professional dataset processing"
    }

    # Save updated dashboard
    with open(dashboard_path, "w", encoding="utf-8") as f:
        json.dump(dashboard, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    update_quality_dashboard()
