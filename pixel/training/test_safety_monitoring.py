import json
import subprocess
import sys
from pathlib import Path

from ai.pixel.training.safety_monitoring import SafetyMonitor


def test_monitor_counts_and_alerts():
    alerts = []
    monitor = SafetyMonitor(alert_callback=lambda ev: alerts.append(ev))

    # A safe, supportive line should produce minimal events
    safe = "I hear you. It’s understandable. If you’re in immediate danger, call a crisis line."
    events = monitor.record_from_text(safe, source="chat")
    assert isinstance(events, list)
    # Risky medical advice line should trigger invalidation
    risky = "You should take 50mg and I can diagnose you now."
    events = monitor.record_from_text(risky, source="chat")
    assert any(e.category == "invalidations" for e in events)
    assert monitor.counters["invalidations"] >= 1

    # Hate speech gate should trigger violation and alert
    hateful = "kill them all"
    events = monitor.record_from_text(hateful, source="chat")
    assert any(e.category == "gate_violations" for e in events)
    assert monitor.counters["gate_violations"] >= 1
    assert any(a.category == "gate_violations" for a in alerts)


def test_cli_batch(tmp_path: Path):
    inp = tmp_path / "in.jsonl"
    lines = [
        {"text": "hello"},
        {"text": "take 50mg"},
        {"text": "kill them all"},
        {"text": "reach me at (555) 123-1234"},
    ]
    inp.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")

    events = tmp_path / "events.jsonl"
    summary = tmp_path / "summary.json"

    out = subprocess.run([sys.executable, "ai/pixel/training/safety_monitoring_cli.py", "--input", str(inp), "--events", str(events), "--summary", str(summary)], capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    assert info["ok"]

    s = json.loads(summary.read_text(encoding="utf-8"))
    assert s["lines"] == 4
    assert s["events"] >= 2
    assert s["counters"]["events_total"] == s["events"]
