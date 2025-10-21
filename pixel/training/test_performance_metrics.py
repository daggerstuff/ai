from pathlib import Path
import json
import sys
import subprocess

from ai.pixel.training.performance_metrics import PerfAggregator


def test_perf_aggregator_basic(tmp_path: Path):
    agg = PerfAggregator()
    # Simulate 10 requests
    for i in range(10):
        agg.record(latency_ms=100 + i * 5, status=200)
    # Add some errors
    agg.record(latency_ms=300, status=500)
    agg.record(latency_ms=250, status=404)

    m = json.loads(agg.export_json())
    assert m["count"] >= 12
    assert m["latency"]["p95"] >= m["latency"]["p50"]
    assert 0.0 <= m["errors"]["rate"] <= 1.0


def test_perf_cli(tmp_path: Path):
    inp = tmp_path / "perf.jsonl"
    rows = [
        {"ts": 1700000000.0, "latency_ms": 120, "status": 200},
        {"ts": 1700000001.0, "latency_ms": 150, "status": 200},
        {"ts": 1700000002.0, "latency_ms": 300, "status": 500},
    ]
    inp.write_text("\n".join(json.dumps(x) for x in rows), encoding="utf-8")

    js = tmp_path / "out.json"
    prom = tmp_path / "out.prom"

    out = subprocess.run([sys.executable, "ai/pixel/training/performance_cli.py", "--input", str(inp), "--json", str(js), "--prom", str(prom)], capture_output=True, text=True, check=True)
    assert js.exists() and prom.exists()
    m = json.loads(js.read_text(encoding="utf-8"))
    assert m["count"] == 3
    pr = prom.read_text(encoding="utf-8")
    assert "pixel_perf_latency_p95_ms" in pr
